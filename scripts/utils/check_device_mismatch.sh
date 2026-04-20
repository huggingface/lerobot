#!/bin/bash

echo "================================================================="
echo "🔍 ĐIỀU TRA LỖI NVIDIA DRIVER MISMATCH TRÊN DI-SERVER"
echo "================================================================="

echo -e "\n[1] PHIÊN BẢN ĐANG CHẠY TRONG KERNEL (Lõi hệ điều hành):"
echo "Đây là phiên bản đã được nạp vào RAM từ lúc máy khởi động."
if [ -f /proc/driver/nvidia/version ]; then
    cat /proc/driver/nvidia/version | grep "NVIDIA"
else
    echo "❌ Không tìm thấy thông tin kernel module. Có thể driver chưa được nạp."
fi

echo -e "\n[2] PHIÊN BẢN THƯ VIỆN TRÊN Ổ CỨNG (User-space Libraries):"
echo "Đây là phiên bản mà PyTorch và nvidia-smi đang cố gắng gọi."
dpkg -l | grep libnvidia-compute | awk '{print $2, $3}'

echo -e "\n[3] LỊCH SỬ CẬP NHẬT GẦN ĐÂY (Thủ phạm gây ra Mismatch):"
echo "Tìm kiếm các bản cập nhật 'nvidia' trong log của hệ thống (dpkg.log)..."
if [ -f /var/log/dpkg.log ]; then
    grep -i "nvidia" /var/log/dpkg.log | grep " upgrade " | tail -n 5
else
    echo "❌ Không có quyền đọc /var/log/dpkg.log hoặc file không tồn tại."
fi

echo -e "\n================================================================="
