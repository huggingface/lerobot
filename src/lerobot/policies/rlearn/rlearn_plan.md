## General Value/Reward Learning:

I want to implement a general/universal vision and language value function or reward model for robotics/video tasks. Also called a video language conditioned reward model. Integrated with already existing LeRobot code if convenient, use the LeRobot Dataset for dataset and store the reward for a frame in the lerobot frame itself.

Inspired by these papers:

- ReWiND; https://arxiv.org/pdf/2505.10911 (Most applicable and main paper I want to implement ideas from) and code: https://github.com/lucidrains/rewind-reward-pytorch
- LIV; https://arxiv.org/pdf/2306.00958 (Most applicable and 2nd main paper I want to implement ideas from) and code https://github.com/penn-pal-lab/LI
- VLC: Video-Language Critic: Transferable Reward Functions for Language-Conditioned Robotics: https://arxiv.org/pdf/2405.19988 (Most applicable and 3rd paper I want to implement ideas from) and code: https://github.com/minttusofia/video_language_critic

And these papers which are also relevant:

- https://www.dyna.co/dyna-1/research (Main company I want to reproduce the eventual results from)
- vip; https://arxiv.org/pdf/2210.00030
- uvd; https://arxiv.org/pdf/2310.08581
- vlm in context; https://arxiv.org/pdf/2411.04549
- https://www.youtube.com/watch?v=JfZYtpEisoM

Little less relevant but still similar papers:

- Learning Generalizable Robotic Reward Functions from “In-The-Wild” Human Videos,
- XIRL: Cross-embodiment Inverse Reinforcement Learning,
- Video-Language Critic: Transferable Reward https://arxiv.org/pdf/2405.19988
- Functions for Language-Conditioned Robotics,
- LORel, Language-Driven Representation Learning for Robotics https://sites.google.com/view/robotlorel
- RoboCLIP: One Demonstration is Enough to Learn Robot Policies https://arxiv.org/pdf/2310.07899
- Points2Rewards: learn first key points and then uses the keypoints to learn general value function/policy https://semrob.github.io/docs/2025_rss_semrob.github.io_paper20.pdf
- Language-Driven Representation Learning for Robotics: https://arxiv.org/pdf/2302.12766v1
- R3M: A Universal Visual Representation for Robot Manipulation: https://arxiv.org/pdf/2203.12601v3

Input should be the current image or whole video and the task goal specified in text/language. Output is current reward.
Archiutecture:
_ inputs: video o1:T (or current o1:t), language z;
_ DINO v3 ViT-B/16 (86M params): https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m for vision encoding
\_ sentence-transformers/all-MiniLM-L12-v2: https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2 for text encoding \* Temporal module: small causal transformer ("cross-modal sequential aggregator"), with first-frame positional embedding (to avoid position cheating), frame-dropout, and stride sampling; outputs per-timestep logits.

Loss: See this chatgpt thread: https://chatgpt.com/s/t_68999a50a0b081919abc365cdd205e01

Past images: (for example a reward method go to 3rd floor, has to know what floor it was on and what pas actions it did, can we attend or encorperate images of decision from history in one way?) Maybe via this paper: Learning Long-Context Diffusion Policies via Past-Token Prediction

Amount of frames needed for test/generalization: 1M frames? or ~20% of IPEC-COMMUNITY/bc_z_lerobot

Eval:
Implement something like voc score , or ROC rank order correlation between reward leanredna and ev reward from sim, or use something else to do additional evaluation

Ideas:

- Incorporate training on multiple horizons: as in label same dataset for longer horizons: make a sandwich (long), put cheese on bread (medium) and even smaller horizons: go down or close gripper (small)
- Incorporate navigation goals “walk towards the kitchen”, make sure we fix CLIP contrastive learning issue of positional text misunderstanding where model doesnnt learn difference between "horse right of cow" and "horse left of cow" “Move right” potentially train with more other data or even actionable world models such as Genie 3 (https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/)

How to use a general reward model (use cases): - Train rl policy on it - Success detection - Do exploraion - Do task via planning and search to optimize reward - Filter out bad episodes in large datasets from imitation learning

Potential Datasets: (start with dataset that is most clean for this and works best with chosen way of doing evals)
_ Epic-Kitchens-100
_ Something-Something v. 2 Dataset https://www.qualcomm.com/developer/software/something-something-v-2-dataset
_ Ego4D (3000 hours)
_ Open X-Embodiment (OXE)
\_ Agi bot world: https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha

- GalexiAI dataset: https://opengalaxea.github.io/G0/
  _ GTEA+ Gaze: https://cbs.ic.gatech.edu/fpv/
  _ YouCook2 dataset
  \_ HOWTO100M: https://www.di.ens.fr/willow/research/howto100m/
- Genie generated dataset?

### TODOs:

- Implement first architecture [x]
- Implement processors [x]
- Choose right loss metric(s) [x]
- Make dataset with script that generated the dataset (IPEC-COMMUNITY/bc_z_lerobot) ready in lerobot format (and be able to visualize in dataset visualizer)
  - Annotate with ReWiND-style 0→1 progress rewards [x]
  - Visualize to check [x]
- Implement eval score or metric that is robust and can deal with generalization/is a good metric to try different architectures. And use it in an eval jupyter notebook with visalization of the live reward next to the video for part of the dataset: VOC score and score with correct and incorrect language captions [x]
- Do first training [x]
- Implement on-the-fly progress label generation (no need for pre-annotated rewards) [x]
- Try different losses
  - Only rewind loss [x]
  - Exactly similar to: https://github.com/lucidrains/rewind-reward-pytorch/blob/main/rewind_reward_pytorch/rewind_reward.py#L11 [x]
  - Try DINO v2 as encoder Base 86 M: with https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2 [x]
  - Test rewind (evaluate) [x]
- benchmark siglip 2 vs this implementation forward pass, debug speed [x]
- use siglip 2 [x]
- Overfit on one episode []
- Cleanup code? [] + enable language loss
- Convert python -m lerobot.datasets.v21.convert_dataset_v20_to_v21 --repo-id=IPEC-COMMUNITY/bc_z_lerobot and train on 1 percent
- Then on 10 percent []
- Ablation 16 sucessive frame vs 16 frame samples with stride 2 or 4 []
- Add more artificial text to dataset generated by vlm (google gemini) []
  - See google gemini vlm caption [] https://gemini.google.com/app/7e332ffaf32580f2
  - Multiple captions per video, creat method to generate as much data as possible etc [] https://arxiv.org/abs/2508.13446, https://arxiv.org/pdf/2412.04453
- Add other datasets from OXE metioned in rewind []
- Extend evaluation []
- Ablation for size vision encoder, language encoder, temporal head []
- Add other datasets metnioned here []
- How can we improve spatial aware learning? solve issue of Contrastive learning and position []


