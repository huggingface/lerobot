import rerun as rr
from datasets import load_from_disk

# download/load dataset in pyarrow format
print("Loading dataset…")
#dataset = load_dataset("lerobot/aloha_mobile_trossen_block_handoff", split="train")
dataset = load_from_disk("tests/data/aloha_mobile_trossen_block_handoff/train")

# select the frames belonging to episode number 5
print("Select specific episode…")

print("Starting Rerun…")
rr.init("rerun_example_lerobot", spawn=True)

print("Logging to Rerun…")
# for frame_index, timestamp, cam_high, cam_left_wrist, cam_right_wrist, state, action, next_reward in zip(

for d in dataset:
    rr.set_time_sequence("frame_index", d["frame_index"])
    rr.set_time_seconds("timestamp", d["timestamp"])
    rr.log("observation.images.cam_high", rr.Image( d["observation.images.cam_high"]))
    rr.log("observation.images.cam_left_wrist", rr.Image(d["observation.images.cam_left_wrist"]))
    rr.log("observation.images.cam_right_wrist", rr.Image(d["observation.images.cam_right_wrist"]))
    #rr.log("observation/state", rr.BarChart(state))
    #rr.log("observation/action", rr.BarChart(action))
    for idx, val in enumerate(d["action"]):
        rr.log(f"action_{idx}", rr.Scalar(val))

    for idx, val in enumerate(d["observation.state"]):
        rr.log(f"state_{idx}", rr.Scalar(val))