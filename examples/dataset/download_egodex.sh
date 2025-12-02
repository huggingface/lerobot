#!/bin/bash

# Download EgoDex dataset
# Reference: https://arxiv.org/abs/2505.11709, https://github.com/apple/ml-egodex
#
# Usage: ./download_egodex.sh [output_dir] [parts...]
#
# Examples:
#   ./download_egodex.sh ./data test           # Download test set only (16 GB)
#   ./download_egodex.sh ./data part1 part2    # Download training parts 1 and 2
#   ./download_egodex.sh ./data all            # Download everything (~1.7 TB)
#
# Available parts:
#   test   - Test set (16 GB)
#   part1  - Training set part 1 (300 GB)
#   part2  - Training set part 2 (300 GB)
#   part3  - Training set part 3 (300 GB)
#   part4  - Training set part 4 (300 GB)
#   part5  - Training set part 5 (300 GB)
#   extra  - Additional data (200 GB)
#   all    - Download all parts (~1.7 TB total)

set -e

BASE_URL="https://ml-site.cdn-apple.com/datasets/egodex"

# Map part names to filenames
declare -A PART_FILES=(
    ["test"]="test.zip"
    ["part1"]="part1.zip"
    ["part2"]="part2.zip"
    ["part3"]="part3.zip"
    ["part4"]="part4.zip"
    ["part5"]="part5.zip"
    ["extra"]="extra.zip"
)

ALL_PARTS=("test" "part1" "part2" "part3" "part4" "part5" "extra")

usage() {
    echo "Usage: $0 <output_dir> <parts...>"
    echo ""
    echo "Examples:"
    echo "  $0 ./data test           # Download test set only (16 GB)"
    echo "  $0 ./data part1 part2    # Download training parts 1 and 2"
    echo "  $0 ./data all            # Download everything (~1.7 TB)"
    echo ""
    echo "Available parts: test, part1, part2, part3, part4, part5, extra, all"
    exit 1
}

download_part() {
    local output_dir="$1"
    local part="$2"
    local filename="${PART_FILES[$part]}"
    local url="${BASE_URL}/${filename}"
    local output_file="${output_dir}/${filename}"

    echo "----------------------------------------"
    echo "Downloading: ${part} (${filename})"
    echo "URL: ${url}"
    echo "Output: ${output_file}"
    echo "----------------------------------------"

    # Download with curl, showing progress
    curl -L --progress-bar "${url}" -o "${output_file}"

    # Unzip
    echo "Extracting ${filename}..."
    unzip -q "${output_file}" -d "${output_dir}"

    # Optionally remove zip file to save space
    # Uncomment the next line if you want to delete zips after extraction
    # rm "${output_file}"

    echo "Done: ${part}"
    echo ""
}

# Check arguments
if [ $# -lt 2 ]; then
    usage
fi

OUTPUT_DIR="$1"
shift

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Determine which parts to download
PARTS_TO_DOWNLOAD=()

for arg in "$@"; do
    if [ "$arg" == "all" ]; then
        PARTS_TO_DOWNLOAD=("${ALL_PARTS[@]}")
        break
    elif [ -n "${PART_FILES[$arg]}" ]; then
        PARTS_TO_DOWNLOAD+=("$arg")
    else
        echo "Error: Unknown part '${arg}'"
        echo "Available parts: test, part1, part2, part3, part4, part5, extra, all"
        exit 1
    fi
done

if [ ${#PARTS_TO_DOWNLOAD[@]} -eq 0 ]; then
    echo "Error: No valid parts specified"
    usage
fi

echo "========================================"
echo "EgoDex Dataset Download"
echo "========================================"
echo "Output directory: ${OUTPUT_DIR}"
echo "Parts to download: ${PARTS_TO_DOWNLOAD[*]}"
echo "========================================"
echo ""

# Download each part
for part in "${PARTS_TO_DOWNLOAD[@]}"; do
    download_part "${OUTPUT_DIR}" "${part}"
done

echo "========================================"
echo "Download complete!"
echo "Data saved to: ${OUTPUT_DIR}"
echo "========================================"

