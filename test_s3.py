#!/usr/bin/env python3
"""
Test script for S3 integration with LeRobot datasets.
Tests loading the robodata/airoa-moma dataset from S3.
"""

import sys
import os
from upath import UPath as Path
from dotenv import load_dotenv

load_dotenv()

# Add lerobot to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

s3_path = "s3://d-gigachat-vision/robodata/airoa-moma"
print(Path(s3_path))
print("OOOOK")

access_key_id = os.getenv('AAAA')
print(access_key_id)

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

meta_data = LeRobotDatasetMetadata(
    repo_id="robodata/airoa-moma",
    root=s3_path,
    revision="main",
)

