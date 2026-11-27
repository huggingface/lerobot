# FLUX 3 Action

FLUX 3 Action is a 7B world action model. It takes camera frames, the robot's state and a text instruction, and returns the next 32 actions, denoised together with the next 32 frames. Fine-tuned on DROID, it places first on the RoboLab-120 benchmark at 42.6% task success. The same weights fine-tune to a new robot, a simulator or a video game with a dataset module and a config.

Technical Report on design decisions and training: XXX
Model Card: https://huggingface.co/black-forest-labs/FLUX-3-action
Data preparation, training recipes and inference code: https://github.com/black-forest-labs/flux-action
