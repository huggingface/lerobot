from lerobot.processor.observation_processor import VanillaObservationProcessorStep
from lerobot.processor.converters import create_transition
from lerobot.processor import TransitionKey
from lerobot.utils.constants import OBS_IMAGE
import numpy as np

processor = VanillaObservationProcessorStep()

# Create a mock image (H, W, C) format, uint8
image = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)

observation = {"pixels": image}
transition = create_transition(observation=observation)
breakpoint()
result = processor(transition)
processed_obs = result[TransitionKey.OBSERVATION]

# Check that the image was processed correctly
assert OBS_IMAGE in processed_obs
processed_img = processed_obs[OBS_IMAGE]
