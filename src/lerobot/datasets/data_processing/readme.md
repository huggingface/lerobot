# Dataset Processing Utilities

This directory contains utilities for processing, downloading, and managing LeRobot datasets. These tools facilitate dataset manipulation including downloading from HuggingFace Hub, video clipping, data processing, and uploading refined datasets.

## Files Overview

### [`download_dataset_locally.py`](download_dataset_locally.py)

**Purpose:** Downloads LeRobot datasets from HuggingFace Hub to local storage.

**Key Functions:**
- `download_keychain_dataset()` - Main download function that fetches datasets from HF Hub
- `get_directory_size()` - Calculates total size of downloaded dataset in MB
- `main()` - Entry point with configurable download settings

**Features:**
- Downloads both data and video files from HuggingFace repositories
- Configurable download location (defaults to `HF_LEROBOT_HOME`)
- Force cache sync option for refreshing local files
- Comprehensive logging to both console and file (`download_log.txt`)
- Dataset statistics display (episodes, frames, FPS, features, camera keys)
- Error handling with detailed logging

**Usage:**
```bash
python download_dataset_locally.py
```

**Configuration:**
- Default repository: `"ywu67/record-test30"`
- Default download directory: `"./downloaded_dataset"`
- Videos included by default
- Logs saved to `download_log.txt`

---

### [`process_data.py`](process_data.py)

**Purpose:** Comprehensive dataset processing script for clipping videos, processing data, and uploading refined datasets to HuggingFace Hub.

**Key Functions:**
- `parse_arguments()` - Command line argument parsing for flexible configuration
- `load_local_data()` - Loads dataset from local files (parquet or JSONL format)
- `clip_split_hf_videos()` - Clips episodes to specified duration and processes videos
- `merge_clipped_videos_to_hub()` - Merges all clipped data and videos into single files
- `upload_to_hf()` - Uploads processed dataset to new HuggingFace repository
- `verify_clipping_results()` - Validates that clipping was performed correctly
- `get_video_duration_ffprobe()` - Gets video duration using ffprobe

**Features:**
- **Data Processing:** Clips episodes to specified duration (default: 10 seconds)
- **Video Processing:** Optional video downloading, clipping, and merging
- **Local/Remote Support:** Works with both HuggingFace Hub and local datasets
- **Verification:** Built-in verification system to validate clipping results
- **Flexible Output:** Configurable output directories and repository IDs
- **Metadata Preservation:** Updates and preserves dataset metadata throughout processing
- **Error Handling:** Robust error handling with detailed logging

**Command Line Arguments:**
```bash
python process_data.py [OPTIONS]
```

**Key Options:**
- `--repo_id`: Source HuggingFace repository ID
- `--clip_second`: Duration to clip from each episode (default: 10.0)
- `--output_dir`: Directory for clipped dataset files
- `--refined_dir`: Directory for merged dataset files
- `--new_repo_id`: Target HuggingFace repository for upload
- `--process_videos`: Enable video processing (default: False)
- `--local_data_path`: Use local dataset instead of downloading
- `--verify_clipping`: Enable verification of clipping results

**Processing Pipeline:**
1. Load dataset (from HF Hub or local files)
2. Clip episodes to specified duration
3. Process and clip videos (if enabled)
4. Merge all clipped data into single files
5. Upload to new HuggingFace repository
6. Verify results (if requested)

---

### [`read_hf_datasets.py`](read_hf_datasets.py)

**Purpose:** Analyzes and reads downloaded LeRobot datasets, providing comprehensive statistics and information.

**Key Functions:**
- Loads dataset metadata from `info.json`
- Reads aggregated statistics from `stats.json`
- Processes all parquet data files
- Analyzes episodes and calculates statistics

**Features:**
- **Dataset Overview:** Displays total episodes, frames, FPS, and duration
- **Episode Analysis:** Detailed breakdown of each episode including:
  - Number of frames per episode
  - Duration of each episode
  - Calculated FPS per episode
- **File Discovery:** Automatically finds and processes all parquet files
- **Statistics Display:** Shows aggregated dataset statistics if available

**Output Information:**
- Total episodes and frames
- Dataset FPS and total duration
- Per-episode frame counts and durations
- Calculated FPS validation
- Aggregated statistics (if available)

**Usage:**
```bash
python read_hf_datasets.py
```

**Expected Directory Structure:**
```
downloaded_dataset/
├── meta/
│   ├── info.json          # Required: Dataset metadata
│   └── stats.json         # Optional: Aggregated statistics
└── data/
    └── **/*.parquet       # Dataset files
```

---

## Workflow Overview

The typical dataset processing workflow using these utilities:

1. **Download Dataset**
   ```bash
   python download_dataset_locally.py
   ```
   Downloads dataset from HuggingFace Hub to local storage.

2. **Analyze Dataset**
   ```bash
   python read_hf_datasets.py
   ```
   Examines downloaded dataset structure and statistics.

3. **Process Dataset**
   ```bash
   python process_data.py --process_videos --verify_clipping
   ```
   Clips episodes, processes videos, and uploads refined dataset.

## Requirements

### Dependencies
- `pandas` - Data manipulation and analysis
- `datasets` - HuggingFace datasets library
- `huggingface_hub` - HuggingFace Hub integration
- `moviepy` - Video processing capabilities
- `pathlib` - Path handling
- `json` - JSON data processing
- `ffprobe` - Video duration analysis

### External Tools
- `ffprobe` (part of FFmpeg) - Required for video duration verification

## Configuration Notes

### Camera Setup
- Maintain consistent resolution across both cameras (left/top)
- Supported camera views: `"left"` and `"top"`

### Video Processing
- Video processing is optional (controlled by `--process_videos` flag)
- Videos are clipped and merged while preserving quality
- Supports MP4 format with H.264 codec

### File Formats
- **Data:** Parquet files for efficient storage and processing
- **Videos:** MP4 files with H.264 encoding
- **Metadata:** JSON format for dataset information

## Error Handling

All scripts include comprehensive error handling:
- Detailed logging with timestamps
- Graceful handling of missing files/directories
- Validation of required dependencies
- Rollback capabilities for failed operations
- Clear error messages and debugging information

## Output Structure

### Clipped Dataset (`clipped_dataset/`)
```
clipped_dataset/
├── data/chunk-000/
│   └── file-{episode:03d}.parquet
└── videos/{camera}/chunk-000/
    └── file-{episode:03d}.mp4
```

### Refined Dataset (`refined_dataset/`)
```
refined_dataset/
├── dataset/                    # HuggingFace format
├── meta/info.json             # Updated metadata
└── videos/{camera}/chunk-000/
    └── file-000-new.mp4       # Merged video