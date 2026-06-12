# Depth Context: LeRobot Target Annotator

A real-time visual goal conditioning tool for the LeRobot project, enabling open-vocabulary object targeting via YOLO-World and stabilized bounding box overlays.

## Document Summaries (Local Files)
| File | Role & Significance (Summary) | Freshness | Status | Relationship |
| :--- | :--- | :--- | :--- | :--- |
| `README.md` | **[Primary SSoT]** Comprehensive guide covering visual goal conditioning, stabilization design, camera convention, state persistence, and installation. | 2026-05-08 | ACTIVE | - |
| `annotate_stream.py` | **[Core Script]** Main implementation — YOLO-World detection, IoU-gated stabilization buffer, ZMQ publisher, operator HUD, state persistence (save/restore zones + frozen bbox). | 2026-05-08 | ACTIVE | - |
| `probe_cameras.py` | **[Utility]** Pre-flight camera probe — tiles all connected cameras into a single window with index labels for quick visual identification. | 2026-05-08 | ACTIVE | - |
| `annotate_stream_state.json` | **[Config]** Auto-generated state file (camera index, classes, zones, frozen bbox). Created on first quit; restored on next launch unless `--fresh`. Git-ignored — not committed. | 2026-05-08 | ACTIVE | - |
| `requirements.txt` | **[Config]** Project dependencies (ultralytics, pyzmq). | 2026-04-28 | ACTIVE | - |
| `LICENSE` | **[Legal]** MIT License. | 2026-04-28 | ACTIVE | - |

## Sub-Folder Index
| Folder | Purpose & Strategy | Key SSoT | Depth |
| :--- | :--- | :--- | :--- |
| `archive/` | Superseded script versions (v2, v4). Not used at runtime. | - | 0f / depth:0 |
| `examples/` | Usage examples and shell scripts for integrating with `lerobot-record`. | `record_command.sh` | 1f / depth:0 |
| `snapshots/` | HUD snapshots saved with `S` key. Git-ignored. | - | 0f / depth:0 |

## Camera Convention
| Index | Camera | Role |
|-------|--------|------|
| 0 | Gripper / wrist | OpenCV → `front` in lerobot-record |
| 1 | Logitech C920 (overview) | Annotated by this tool → ZMQ → `annotated` in lerobot-record |

Run `python probe_cameras.py` to verify before recording.

## Local Audit Logic
- **Annotation Integrity:** The ZMQ stream must match the LeRobot `ZMQCamera` protocol for seamless recording.
- **Stabilization:** Buffer parameters in `annotate_stream.py` should be tuned to balance commit speed and tracking robustness.
- **State Persistence:** `annotate_stream_state.json` is auto-managed; `--fresh` skips restore. Delete the file to reset entirely.
- **Progressive Audit:** Use `hcm_audit.py` to maintain table integrity across the branch nodes.
