#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path

import pyarrow.parquet as pq


def _copy_or_symlink(src: Path, dst: Path, *, symlink: bool) -> None:
    if not src.exists():
        return
    if symlink:
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
    elif src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _patch_data_files(src_root: Path, dst_root: Path, action_key: str) -> tuple[int, int]:
    src_data_dir = src_root / "data"
    dst_data_dir = dst_root / "data"
    dst_data_dir.mkdir(parents=True, exist_ok=True)

    num_files = 0
    num_rows = 0

    for src_file in sorted(src_data_dir.rglob("*.parquet")):
        rel = src_file.relative_to(src_data_dir)
        dst_file = dst_data_dir / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        table = pq.read_table(src_file)
        if action_key not in table.column_names:
            raise KeyError(f"'{action_key}' が parquet に見つかりません: {src_file}")

        action_column = table[action_key]
        if "action" in table.column_names:
            action_idx = table.schema.get_field_index("action")
            table = table.set_column(action_idx, "action", action_column)
        else:
            table = table.append_column("action", action_column)

        pq.write_table(table, dst_file)
        num_files += 1
        num_rows += table.num_rows

    return num_files, num_rows


def _patch_meta(dst_root: Path, action_key: str) -> None:
    meta_dir = dst_root / "meta"
    info_path = meta_dir / "info.json"
    stats_path = meta_dir / "stats.json"

    info = json.loads(info_path.read_text())
    if action_key not in info["features"]:
        raise KeyError(f"info.json の features に '{action_key}' がありません")
    info["features"]["action"] = deepcopy(info["features"][action_key])
    info["features"]["action"]["description"] = (
        f"Alias for PI0.5 training. Copied from '{action_key}'."
    )

    data_size_mb = sum(f.stat().st_size for f in (dst_root / "data").rglob("*.parquet")) / (1024 * 1024)
    info["data_files_size_in_mb"] = round(data_size_mb, 2)
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")

    stats = json.loads(stats_path.read_text())
    if action_key in stats:
        stats["action"] = deepcopy(stats[action_key])
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AIRoA MoMa の action キーを PI0.5 用に 'action' として追加した学習用コピーを作成します。"
    )
    parser.add_argument("--src_root", type=Path, required=True, help="元データセット root")
    parser.add_argument("--dst_root", type=Path, required=True, help="出力データセット root")
    parser.add_argument(
        "--action_key",
        type=str,
        default="action.absolute",
        help="PI0.5 が参照する 'action' にコピーする元キー",
    )
    parser.add_argument("--force", action="store_true", help="dst_root が存在する場合に削除して再生成")
    parser.add_argument(
        "--symlink_videos",
        action="store_true",
        help="videos ディレクトリをコピーせずシンボリックリンクにする",
    )
    args = parser.parse_args()

    src_root = args.src_root.expanduser().resolve()
    dst_root = args.dst_root.expanduser().resolve()

    if not (src_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"LeRobot 形式の meta/info.json が見つかりません: {src_root}")

    if dst_root.exists():
        if not args.force:
            raise FileExistsError(f"{dst_root} は既に存在します。上書きする場合は --force を付けてください。")
        shutil.rmtree(dst_root)

    dst_root.mkdir(parents=True, exist_ok=True)

    # Copy metadata and optional repository files.
    shutil.copytree(src_root / "meta", dst_root / "meta")
    if (src_root / "README.md").exists():
        shutil.copy2(src_root / "README.md", dst_root / "README.md")
    if (src_root / ".gitattributes").exists():
        shutil.copy2(src_root / ".gitattributes", dst_root / ".gitattributes")

    # Copy/symlink videos first (not modified).
    _copy_or_symlink(src_root / "videos", dst_root / "videos", symlink=args.symlink_videos)

    num_files, num_rows = _patch_data_files(src_root, dst_root, args.action_key)
    _patch_meta(dst_root, args.action_key)

    print("=== PI0.5 dataset prepare completed ===")
    print(f"src_root: {src_root}")
    print(f"dst_root: {dst_root}")
    print(f"action alias: action <- {args.action_key}")
    print(f"processed parquet files: {num_files}")
    print(f"processed rows: {num_rows}")
    print(f"videos mode: {'symlink' if args.symlink_videos else 'copied'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
