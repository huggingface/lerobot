import base64
import os
from openai import OpenAI

# Helper function to convert local image to data URL
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

client = OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key="fw_M9dJXroS7AvXgmaBetAiTL")

# Common path variable
base_path = "/content/drive/MyDrive/harezmi-extend-dump/vlm_probe_task0/vlm_inputs/frame_0006"

images_metadata = [
    ("reference.target.task_7_milk", f"{base_path}/00_reference.target.task_7.00.task_7_milk.png"),
    ("temporal.scene.prev_2", f"{base_path}/01_temporal.scene.prev_2.observation.images.image.png"),
    ("temporal.scene.prev_1", f"{base_path}/02_temporal.scene.prev_1.observation.images.image.png"),
    ("temporal.scene.current", f"{base_path}/03_temporal.scene.current.observation.images.image.png"),
]

prompt = """
Task: pick up the milk and place it in the basket.

You are given images in this order:
1. reference.target.task_7_milk: target object reference only.
2. temporal.scene.prev_2: earlier whole-scene state.
3. temporal.scene.prev_1: later whole-scene state.
4. temporal.scene.current: current whole-scene state to score.

Score only the final current image. Use the previous scene images only to understand motion/progress.
Use the reference image only to identify the milk carton. Do not reward movement toward other objects.

Return JSON only:
{"score": 0.0, "reason": "short reason"}

Score guide:
0.0 = milk is visible but no progress toward grasping or basket placement
0.2 = gripper is closer to or aligned with the milk
0.4 = gripper is touching or about to grasp the milk
0.6 = milk is grasped or lifted
0.8 = milk is near or above the basket
1.0 = milk is inside the basket
"""

content = []
for label, path in images_metadata:
    # Convert local file to base64 data URL so the model can process it
    base64_image = encode_image(path)
    
    content.append({"type": "text", "text": f"Image label: {label}"})
    content.append({
        "type": "image_url", 
        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
    })

content.append({"type": "text", "text": prompt})

messages = [{"role": "user", "content": content}]

chat_response = client.chat.completions.create(
    model="accounts/fireworks/models/qwen3-vl-30b-a3b-thinking",
    messages=messages,
    temperature=0.6,
    top_p=0.95,
    presence_penalty=0.0,
    extra_body={
      "top_k": 20,
      "min_p": 0.0,
      "repetition_penalty":1.0,
    },
)

print("Chat response:", chat_response.choices[0].message.content)
