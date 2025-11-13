#!/usr/bin/env python3
"""
RAG for Robot Dataset Health — Episode Motor Averages (conversational)

What this script does
---------------------
1) Loads per-episode 10s motor averages (motor-major JSON: motor -> {episode_id: value|list|str})
2) Computes per-motor μ/σ, z-scores per episode, outlier flags, health_score
3) Builds episode "doc" text blobs for Cohere Embed + Rerank
4) Indexes embeddings in FAISS
5) Optional conversational summaries via Cohere Command-R
6) CLI:
   - summary                  : build index (if needed) + print health summary
   - query   --q "..."        : semantic search (Embed->ANN->Rerank)
   - similar --ep N           : episodes similar to ep_N
   - explain --ep N [--motor] : WHY outlier + side-by-side PNG

Inputs
------
--avg-json      : path to averages JSON (motor -> {ep: float|list|str})
--frames-json   : (optional) ep_id -> [first-frame paths or URLs]
--repo-dir      : directory to store artifacts (docs.jsonl, stats.json, faiss.index, embeddings.npy)

Install
-------
pip install cohere faiss-cpu numpy pillow tqdm

Env
---
export COHERE_API_KEY=YOUR_KEY

# 1) Build + conversational summary
python rag_robot_health.py \
  --avg-json initpos_stats/Sahana16/record-medicine/first10s_avg.json \
  --repo-dir Sahana16/record-medicine \
 --chat --style direct summary

# 2) Conversational query
python rag_robot_health.py \
  --avg-json initpos_stats/Sahana16/record-medicine/first10s_avg.json \
  --repo-dir record-medicine \
  --chat --style friendly \
  query --q "outliers on elbow_flex"

"""

from __future__ import annotations
import os
import json
import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw

import faiss
import cohere


# ---------------------------- Config ----------------------------
MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper_pos",
]

OUTLIER_Z = 3.0
CANDIDATE_Z = 2.0
RERANK_TOP_K = 12
ANN_CAND_K = 64
EMBED_MODEL = "embed-english-v3.0"
RERANK_MODEL = "rerank-english-v3.0"
CHAT_MODEL  = "command-r"

# ---------------------------- Data types ----------------------------
@dataclass
class EpisodeDoc:
    ep_id: str
    split: str | None
    task: str | None
    motor_avgs: Dict[str, float]
    motor_mu: Dict[str, float]
    motor_sigma: Dict[str, float]
    motor_z: Dict[str, float]
    outlier_flags: Dict[str, bool]
    health_score: float
    first_frames: List[str] | None

    def to_text_blob(self) -> str:
        def fmt(v):
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                return "nan"
            return f"{v:.6f}"
        parts = []
        parts.append(f"ep_id: {self.ep_id}")
        if self.task: parts.append(f"task: {self.task}")
        if self.split: parts.append(f"split: {self.split}")
        parts.append("motor_avgs: {" + ", ".join(f"{m}:{fmt(self.motor_avgs.get(m, float('nan')))}" for m in MOTORS) + "}")
        parts.append("motor_z: {"    + ", ".join(f"{m}:{fmt(self.motor_z.get(m, float('nan')))}" for m in MOTORS) + "}")
        parts.append("outliers: {"   + ", ".join(f"{m}:{'true' if self.outlier_flags.get(m, False) else 'false'}" for m in MOTORS) + "}")
        parts.append(f"health_score: {fmt(self.health_score)}")
        return "; ".join(parts)

# ---------------------------- IO helpers ----------------------------
def _looks_num_str(s: str) -> bool:
    s = s.strip()
    if not s: return False
    if s[0] in ('+', '-'): s = s[1:]
    return s.replace(".", "", 1).isdigit()

def _coerce_scalar(val) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        return float(val.strip()) if _looks_num_str(val) else float("nan")
    if isinstance(val, (list, tuple)):
        nums = []
        for x in val:
            if isinstance(x, (int, float)): nums.append(float(x))
            elif isinstance(x, str) and _looks_num_str(x): nums.append(float(x))
        return float(np.mean(nums)) if nums else float("nan")
    return float("nan")

def load_avg_json(path: Path) -> Dict[str, Dict[str, float]]:
    """
    Accepts motor-major JSON with floats/strings/lists.
    Returns: motor -> {episode_id: float_mean}
    """
    raw = json.loads(Path(path).read_text())
    filtered: Dict[str, Dict[str, float]] = {}
    bad = []
    for m in MOTORS:
        motor_map = {}
        for ep, val in (raw.get(m, {}) or {}).items():
            v = _coerce_scalar(val)
            if math.isnan(v):
                bad.append((m, ep, val))
            motor_map[str(ep)] = v
        filtered[m] = motor_map
    if bad:
        print(f"[warn] {len(bad)} invalid entries; set to NaN. First few:")
        for b in bad[:5]:
            print("   motor:", b[0], "episode:", b[1], "value:", str(b[2])[:120])
    return filtered

def load_frames_json(path: Path | None) -> Dict[str, List[str]]:
    if not path: return {}
    return {str(k): v for k, v in json.loads(Path(path).read_text()).items()}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------- Stats & Docs ----------------------------
def compute_mu_sigma(avg_motor_major: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    mu, sigma = {}, {}
    for m in MOTORS:
        vals = np.array(list(avg_motor_major.get(m, {}).values()), dtype=np.float64)
        if vals.size == 0:
            mu[m], sigma[m] = float("nan"), float("nan")
        else:
            mu[m] = float(np.nanmean(vals))
            sigma[m] = float(np.nanstd(vals, ddof=1) if vals.size > 1 else 0.0)
    return mu, sigma

def build_episode_table(avg_motor_major: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    episodes = set()
    for m in MOTORS:
        episodes.update(avg_motor_major.get(m, {}).keys())
    table = {ep: {} for ep in episodes}
    for m in MOTORS:
        for ep, val in avg_motor_major.get(m, {}).items():
            table[ep][m] = val  # already coerced
    return table

def zscore(value: float, mu: float, sigma: float) -> float:
    if sigma is None or sigma == 0 or math.isnan(sigma):
        return float("nan")
    return (value - mu) / sigma

def health_score_from_z(zs: Dict[str, float]) -> float:
    abs_vals = [abs(v) for v in zs.values() if not math.isnan(v)]
    if not abs_vals: return float("nan")
    return float(1.0 / (1.0 + (sum(abs_vals) / max(1, len(abs_vals)))))

def build_docs(
    avg_motor_major: Dict[str, Dict[str, float]],
    frames_map: Dict[str, List[str]] | None = None,
    global_task: str | None = None,
    global_split: str | None = None
) -> Tuple[List[EpisodeDoc], Dict[str, float], Dict[str, float]]:
    mu, sigma = compute_mu_sigma(avg_motor_major)
    ep_table = build_episode_table(avg_motor_major)
    docs: List[EpisodeDoc] = []
    for ep_id, motor_avgs in ep_table.items():
        motor_z = {m: zscore(motor_avgs.get(m, float("nan")), mu.get(m, float("nan")), sigma.get(m, float("nan"))) for m in MOTORS}
        flags = {m: (abs(z) > OUTLIER_Z) if not math.isnan(z) else False
                 for m in MOTORS
                 for z in [motor_z.get(m, float("nan"))]}
        health = health_score_from_z(motor_z)
        frames = (frames_map or {}).get(str(ep_id))
        docs.append(EpisodeDoc(
            ep_id=str(ep_id),
            split=global_split,
            task=global_task,
            motor_avgs={m: motor_avgs.get(m, float("nan")) for m in MOTORS},
            motor_mu=mu,
            motor_sigma=sigma,
            motor_z=motor_z,
            outlier_flags=flags,
            health_score=health,
            first_frames=frames
        ))
    return docs, mu, sigma

# ---------------------------- Cohere & FAISS ----------------------------
def get_cohere() -> cohere.Client:
    key = os.environ.get("COHERE_API_KEY")
    if not key:
        raise RuntimeError("Set COHERE_API_KEY environment variable.")
    return cohere.Client(key)

def embed_texts_cohere(texts: List[str], input_type: str) -> np.ndarray:
    co = get_cohere()
    out = co.embed(texts=texts, model=EMBED_MODEL, input_type=input_type)
    embs = np.array(out.embeddings, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    return embs / norms

def build_faiss(embs: np.ndarray) -> faiss.IndexFlatIP:
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine via dot on normalized vectors
    index.add(embs)
    return index

def ann_search(index: faiss.IndexFlatIP, q_vec: np.ndarray, k: int = ANN_CAND_K) -> Tuple[np.ndarray, np.ndarray]:
    D, I = index.search(q_vec[None, :], k)
    return I[0], D[0]

def rerank(co: cohere.Client, query: str, doc_texts: List[str], k: int = RERANK_TOP_K) -> List[int]:
    rr = co.rerank(model=RERANK_MODEL, query=query, documents=doc_texts, top_n=min(k, len(doc_texts)))
    return [r.index for r in rr.results]

# ---------------------------- Conversational layer ----------------------------
def chatify(prompt: str, user_context: str = "", style: str = "direct") -> str:
    """
    Use Cohere Command-R to turn structured findings into a concise, conversational blurb.
    """
    co = get_cohere()
    sys = (
        "You are an expert robotics dataset analyst. "
        "Be direct, clear, and actionable in 4-8 sentences. "
        "If listing steps, keep them short and specific."
    )
    full_prompt = (
        f"{prompt}\n\n"
        f"Tone: {style}.\n"
        f"Context:\n{user_context}\n"
        f"Respond as one concise paragraph or a few short bullets."
    )
    try:
        resp = co.chat(model=CHAT_MODEL, message=full_prompt)
        return (resp.text or "").strip()
    except Exception as e:
        return f"(chat unavailable: {e})"

# ---------------------------- Reports ----------------------------
def dataset_health_summary(docs: List[EpisodeDoc]) -> Dict[str, Any]:
    motor_outliers = {m: 0 for m in MOTORS}
    for d in docs:
        for m in MOTORS:
            if d.outlier_flags.get(m, False):
                motor_outliers[m] += 1
    ranked = [d for d in docs if not math.isnan(d.health_score)]
    ranked.sort(key=lambda x: x.health_score)  # low is worse
    worst = ranked[:min(5, len(ranked))]
    best  = list(reversed(ranked))[:min(5, len(ranked))]
    return {
        "motor_outlier_counts": motor_outliers,
        "worst_eps": [(d.ep_id, round(d.health_score, 4)) for d in worst],
        "best_eps":  [(d.ep_id, round(d.health_score, 4)) for d in best],
    }

def print_summary(summary: Dict[str, Any]):
    print("\n== Dataset Health Summary ==")
    print(f"Outlier counts per motor (|z| > {OUTLIER_Z:.1f}):")
    for m in MOTORS:
        print(f"  - {m:14s}: {summary['motor_outlier_counts'][m]}")
    print("\nWorst episodes by health_score:")
    for ep, sc in summary["worst_eps"]:
        print(f"  - ep {ep:>6s}: {sc:.4f}")
    print("\nBest episodes by health_score:")
    for ep, sc in summary["best_eps"]:
        print(f"  - ep {ep:>6s}: {sc:.4f}")
    print()

# ---------------------------- Visual Explain ----------------------------
def top_outlier_motors(doc, top_k=3):
    items = []
    for m in MOTORS:
        z = doc["motor_z"].get(m, float("nan"))
        if math.isnan(z): continue
        items.append((m, z,
                      doc["motor_mu"].get(m, float("nan")),
                      doc["motor_sigma"].get(m, float("nan")),
                      doc["motor_avgs"].get(m, float("nan"))))
    items.sort(key=lambda x: abs(x[1]), reverse=True)
    return items[:top_k]

def find_healthy_exemplars(docs_json, embs, index, target_idx, focus_motor=None, k=3):
    I, _ = ann_search(index, embs[target_idx], k=min(256, len(docs_json)))
    candidates = [i for i in I if i != target_idx]
    def ok(d):
        if focus_motor:
            z = d["motor_z"].get(focus_motor)
            return (z is not None) and (not math.isnan(z)) and (abs(z) < 1.0)
        zs = [abs(v) for v in d["motor_z"].values() if not math.isnan(v)]
        return zs and (sum(zs)/len(zs) < 1.0)
    healthy = [i for i in candidates if ok(docs_json[i])]
    return healthy[:k] if healthy else candidates[:k]

def load_first_images(paths, max_per_ep=2, target_size=(512, 384)):
    imgs = []
    for p in (paths or [])[:max_per_ep]:
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail(target_size)
            imgs.append(im)
        except Exception:
            continue
    return imgs

def make_side_by_side(target_imgs, exemplar_imgs_list, caption, out_png_path):
    pad = 16
    col_w = max([img.width for img in (target_imgs + sum(exemplar_imgs_list, []))] + [320])
    row_h = sum(img.height for img in target_imgs) + (len(target_imgs)-1)*pad if target_imgs else 240
    rows = max(1, max((len(x) for x in exemplar_imgs_list), default=1))
    right_cols = len(exemplar_imgs_list)
    width = pad + col_w + pad + right_cols*(col_w + pad) + pad
    height = pad + max(row_h, rows*((target_imgs[0].height if target_imgs else 240) + pad)) + 120
    canvas = Image.new("RGB", (width, height), (255,255,255))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, pad), caption, fill=(0,0,0))
    # target (left)
    y = pad + 32; x = pad
    for img in target_imgs:
        canvas.paste(img, (x, y)); y += img.height + pad
    # exemplars (right)
    base_x = pad + col_w + pad
    for col, imgs in enumerate(exemplar_imgs_list):
        y = pad + 32; x = base_x + col*(col_w + pad)
        for img in imgs:
            canvas.paste(img, (x, y)); y += img.height + pad
    canvas.save(out_png_path)
    return out_png_path

# ---------------------------- CLI actions ----------------------------
def ensure_repo(docs: List[EpisodeDoc], repo_dir: Path):
    ensure_dir(repo_dir)
    docs_path = repo_dir / "docs.jsonl"
    stats_path = repo_dir / "stats.json"
    embs_path = repo_dir / "embeddings.npy"
    index_path = repo_dir / "faiss.index"

    with docs_path.open("w") as f:
        for d in docs:
            rec = asdict(d)
            rec["text_blob"] = d.to_text_blob()
            f.write(json.dumps(rec) + "\n")

    texts = [d.to_text_blob() for d in docs]
    embs = embed_texts_cohere(texts, input_type="search_document")
    np.save(embs_path, embs)

    index = build_faiss(embs)
    faiss.write_index(index, str(index_path))

    stats = {
        "n_docs": len(docs),
        "motors": MOTORS,
        "embed_model": EMBED_MODEL,
        "rerank_model": RERANK_MODEL,
        "outlier_z": OUTLIER_Z,
        "candidate_z": CANDIDATE_Z,
    }
    stats_path.write_text(json.dumps(stats, indent=2))

    print(f"[repo] wrote {docs_path}")
    print(f"[repo] wrote {embs_path}")
    print(f"[repo] wrote {index_path}")
    print(f"[repo] wrote {stats_path}")

def load_repo(repo_dir: Path) -> Tuple[List[Dict[str, Any]], np.ndarray, faiss.IndexFlatIP]:
    docs_path = repo_dir / "docs.jsonl"
    embs_path = repo_dir / "embeddings.npy"
    index_path = repo_dir / "faiss.index"
    docs_json = [json.loads(l) for l in docs_path.read_text().splitlines() if l.strip()]
    embs = np.load(embs_path)
    index = faiss.read_index(str(index_path))
    return docs_json, embs, index

def cmd_summary(args):
    avg_mm = load_avg_json(args.avg_json)
    frames_map = load_frames_json(args.frames_json) if args.frames_json else {}
    docs, mu, sigma = build_docs(avg_mm, frames_map, global_task=args.task, global_split=args.split)
    ensure_repo(docs, args.repo_dir)
    summary = dataset_health_summary(docs)
    print_summary(summary)

    if args.chat:
        counts = "\n".join([f"{m}: {summary['motor_outlier_counts'][m]}" for m in MOTORS])
        worst = ", ".join([f"ep {e} (score {s})" for e, s in summary["worst_eps"]])
        best  = ", ".join([f"ep {e} (score {s})" for e, s in summary["best_eps"]])
        ctx = (
            f"Outlier counts:\n{counts}\n\n"
            f"Worst episodes: {worst}\nBest episodes: {best}\n"
            f"Outlier threshold: |z| > {OUTLIER_Z}"
        )
        prompt = ("Summarize dataset health for robotics training. "
                  "Explain implications and give 2–3 concrete next steps.")
        print(chatify(prompt, ctx, style=args.style))

def cmd_query(args):
    if not (args.repo_dir / "docs.jsonl").exists():
        avg_mm = load_avg_json(args.avg_json)
        frames_map = load_frames_json(args.frames_json) if args.frames_json else {}
        docs, mu, sigma = build_docs(avg_mm, frames_map, global_task=args.task, global_split=args.split)
        ensure_repo(docs, args.repo_dir)

    docs_json, embs, index = load_repo(args.repo_dir)
    texts = [d["text_blob"] for d in docs_json]

    co = get_cohere()
    q_emb = embed_texts_cohere([args.q], input_type="search_query")[0]
    idxs, _ = ann_search(index, q_emb, k=min(ANN_CAND_K, len(texts)))
    cands = [texts[i] for i in idxs]
    order = rerank(co, args.q, cands, k=min(RERANK_TOP_K, len(cands)))
    final_ids = [idxs[i] for i in order]

    print("\n== Query Results ==")
    for rank, i in enumerate(final_ids, 1):
        d = docs_json[i]
        print(f"{rank:2d}. ep {d['ep_id']} | health={d['health_score']:.4f} | outliers={[m for m,f in d['outlier_flags'].items() if f]}")
        if d.get("first_frames"):
            print(f"    frames: {', '.join(d['first_frames'][:3])}")
    print()

    if args.chat:
        top_blobs = "\n\n".join([docs_json[i]["text_blob"] for i in final_ids[:5]])
        prompt = (f"User question: {args.q}\n"
                  "Answer conversationally for a robotics dataset engineer. "
                  "Reference episode IDs and motors likely at fault. Suggest next checks.")
        print(chatify(prompt, top_blobs, style=args.style))

def cmd_similar(args):
    if not (args.repo_dir / "docs.jsonl").exists():
        avg_mm = load_avg_json(args.avg_json)
        frames_map = load_frames_json(args.frames_json) if args.frames_json else {}
        docs, mu, sigma = build_docs(avg_mm, frames_map, global_task=args.task, global_split=args.split)
        ensure_repo(docs, args.repo_dir)

    docs_json, embs, index = load_repo(args.repo_dir)
    texts = [d["text_blob"] for d in docs_json]
    ep_idx_map = {d["ep_id"]: i for i, d in enumerate(docs_json)}
    if str(args.ep) not in ep_idx_map:
        raise SystemExit(f"ep {args.ep} not found in repo docs")

    i = ep_idx_map[str(args.ep)]
    co = get_cohere()

    q_vec = embs[i]
    idxs, _ = ann_search(index, q_vec, k=min(ANN_CAND_K, len(texts)))
    idxs = [j for j in idxs if j != i]
    cands_text = [texts[j] for j in idxs]
    order = rerank(co, texts[i], cands_text, k=min(RERANK_TOP_K, len(cands_text)))
    final_ids = [idxs[j] for j in order]

    print(f"\n== Episodes similar to ep {args.ep} ==")
    for rank, j in enumerate(final_ids, 1):
        d = docs_json[j]
        print(f"{rank:2d}. ep {d['ep_id']} | health={d['health_score']:.4f} | outliers={[m for m,f in d['outlier_flags'].items() if f]}")
        if d.get("first_frames"):
            print(f"    frames: {', '.join(d['first_frames'][:3])}")
    print()

    if args.chat:
        top_blobs = "\n\n".join([docs_json[j]["text_blob"] for j in final_ids[:5]])
        anchor = docs_json[i]["text_blob"]
        prompt = (f"Compare similar episodes to anchor episode {args.ep}. "
                  "Call out motor dimensions that differ and propose a quick validation checklist.")
        ctx = f"ANCHOR:\n{anchor}\n\nSIMILARS:\n{top_blobs}"
        print(chatify(prompt, ctx, style=args.style))

def cmd_explain(args):
    if not (args.repo_dir / "docs.jsonl").exists():
        avg_mm = load_avg_json(args.avg_json)
        frames_map = load_frames_json(args.frames_json) if args.frames_json else {}
        docs, mu, sigma = build_docs(avg_mm, frames_map, global_task=args.task, global_split=args.split)
        ensure_repo(docs, args.repo_dir)

    docs_json, embs, index = load_repo(args.repo_dir)
    ep_idx_map = {d["ep_id"]: i for i, d in enumerate(docs_json)}
    if str(args.ep) not in ep_idx_map:
        raise SystemExit(f"ep {args.ep} not found in repo docs")

    i = ep_idx_map[str(args.ep)]
    d = docs_json[i]

    offenders = top_outlier_motors(d, top_k=3)
    focus_motor = args.motor if args.motor else (offenders[0][0] if offenders else None)
    ex_ids = find_healthy_exemplars(docs_json, embs, index, i, focus_motor=focus_motor, k=args.k)

    target_imgs = load_first_images(d.get("first_frames"), max_per_ep=2)
    exemplar_imgs_list = [load_first_images(docs_json[j].get("first_frames"), max_per_ep=2) for j in ex_ids]

    out_png = args.repo_dir / f"explain_ep{args.ep}.png"
    cap = f"Episode {args.ep}: outlier vs healthy exemplars (focus motor: {focus_motor})"
    make_side_by_side(target_imgs, exemplar_imgs_list, cap, out_png)

    print(f"\n== Explanation for episode {args.ep} ==")
    if offenders:
        for (m, z, mu, sigma, val) in offenders:
            side = "high" if z > 0 else "low"
            label = " -> OUTLIER" if abs(z) > OUTLIER_Z else (" -> CANDIDATE" if abs(z) > CANDIDATE_Z else "")
            print(f"- {m}: z={z:.2f} ({side}); value={val:.4f}, mean={mu:.4f}, std={sigma:.4f}{label}")
    else:
        print(f"- No strong offenders; health_score={d['health_score']:.4f}")

    print("\nHealthy exemplars shown:")
    for j in ex_ids:
        dj = docs_json[j]
        max_abs_z = max((abs(v) for v in dj['motor_z'].values() if not math.isnan(v)), default=float("nan"))
        print(f"- ep {dj['ep_id']} | health={dj['health_score']:.4f} | max|z|={max_abs_z:.2f}")

    print(f"\nSaved side-by-side: {out_png}\n")

    if args.chat:
        lines = []
        for (m, z, mu, sigma, val) in offenders:
            side = "high" if z > 0 else "low"
            lines.append(f"{m}: z={z:.2f} ({side}), value={val:.4f}, mean={mu:.4f}, std={sigma:.4f}")
        ex_list = ", ".join([docs_json[j]["ep_id"] for j in ex_ids]) if ex_ids else "none"
        ctx = (
            f"Episode {args.ep} offenders:\n" + "\n".join(lines) +
            f"\nExemplars: {ex_list}\nOutlier threshold: |z| > {OUTLIER_Z}\nImage: {out_png}"
        )
        prompt = ("Explain in plain language why this episode is an outlier and "
                  "what the images likely show. Give 2 next steps to confirm/fix.")
        print(chatify(prompt, ctx, style=args.style))

# ---------------------------- Main ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="RAG on motor averages: health + retrieval (conversational)")
    p.add_argument("--avg-json", type=Path, required=True, help="Path to motor-major averages JSON")
    p.add_argument("--frames-json", type=Path, default=None, help="Optional: ep_id -> [first-frame paths]")
    p.add_argument("--repo-dir", type=Path, required=True, help="Directory to store docs/index/embeddings")
    p.add_argument("--task", type=str, default=None, help="Optional task label")
    p.add_argument("--split", type=str, default=None, help="Optional split label (train/eval)")
    p.add_argument("--chat", action="store_true", help="Generate a conversational summary/answer via Cohere")
    p.add_argument("--style", type=str, default="direct", help="Tone: direct | friendly | technical | executive")
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("summary", help="Build index (if needed) and print dataset health summary")
    s1.set_defaults(func=cmd_summary)

    s2 = sub.add_parser("query", help="Text search over episode docs (Embed->ANN->Rerank)")
    s2.add_argument("--q", type=str, required=True)
    s2.set_defaults(func=cmd_query)

    s3 = sub.add_parser("similar", help="Episodes similar to a given episode")
    s3.add_argument("--ep", type=str, required=True)
    s3.set_defaults(func=cmd_similar)

    s4 = sub.add_parser("explain", help="Explain why an episode is an outlier; render images vs healthy exemplars")
    s4.add_argument("--ep", type=str, required=True)
    s4.add_argument("--motor", type=str, default=None, choices=MOTORS + [None])
    s4.add_argument("--k", type=int, default=3, help="Number of healthy exemplars to show")
    s4.set_defaults(func=cmd_explain)

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.func(args)
