# file: sync_data.py
from huggingface_hub import snapshot_download
from pathlib import Path

repo_id = "trietlm0306/vla-4-boxes-2"
# Lưu vào thư mục ngay trong project cho dễ quản lý
local_dir = Path("/home/trietlm/lerobot/data/vla-4-boxes-2")

print(f"⏳ Đang kéo dữ liệu từ {repo_id} về {local_dir}...")

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    # Đảm bảo tải cả các file metadata nhỏ và file video/parquet lớn
    allow_patterns=["*.json", "*.parquet", "*.mp4", "*.md"]
)

print("✅ Hoàn tất! Cấu trúc thư mục hiện tại:")
import os
for root, dirs, files in os.walk(local_dir):
    level = root.replace(str(local_dir), '').count(os.sep)
    indent = ' ' * 4 * (level)
    print(f"{indent}{os.path.basename(root)}/")
    sub_indent = ' ' * 4 * (level + 1)
    for f in files[:3]: # in thử 3 file đầu
        print(f"{sub_indent}{f}")