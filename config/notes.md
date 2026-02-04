# LeRobot - Quick Reference Guide

## Robot Configuration

**Robot ID:** zarax  
**Follower Port:** COM7  
**Leader Port:** COM8  
**Documentation:** https://huggingface.co/docs/lerobot/cameras    

---
## Setup & Environment
### Initial Setup
```bash
cd C:\GitHub\Zaraxxx\lerobot
conda activate lerobot
```
### Hugging Face Login
```bash
huggingface-cli login
```

<br>

## Calibration
### Calibrate Follower (COM7)
```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=COM7 --robot.id=zarax
```
### Calibrate Leader (COM8)
```bash
lerobot-calibrate --teleop.type=so101_leader --teleop.port=COM8 --teleop.id=zarax
```
---
## Camera Configuration
### Find Available Cameras
```bash
lerobot-find-cameras opencv
# or for Intel Realsense cameras:
lerobot-find-cameras realsense
```
<br>

## Teleoperation

### Sans Camera
```bash
lerobot-teleoperate --config_path .\config\teleop\zarax_teleop_config_nocam.yaml
```
### Camera follower uniquement
```bash
lerobot-teleoperate --config_path .\config\teleop\zarax_teleop_config_camdroite.yaml
```

<br>

## Datasets

### Local Dataset Location
```
C:\Users\picip\.cache\huggingface\lerobot\Zarax\
```
### Hub Dataset Location
https://huggingface.co/Zarax/datasets


### Delete Datasets
```bash
python delete_datasets.py --all
# or 
python delete_datasets.py --user Zarax
```
### Recording Datasets 
```bash
# Record 5 episodes
lerobot-record \
  --config_path .\zarax_teleop_config.yaml \
  --repo-id Zarax/zarax-demo \
  --num-episodes 5

# Record 3 episodes
lerobot-record \
  --config_path .\zarax_teleop_config.yaml \
  --repo-id Zarax/zarax-demo \
  --num-episodes 3
```