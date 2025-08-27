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
_ google/siglip2-large-patch16-256: https://huggingface.co/google/siglip2-large-patch16-256 \* Temporal module: small causal transformer (“cross-modal sequential aggregator”), with first-frame positional embedding (to avoid position cheating), frame-dropout, and stride sampling; outputs per-timestep logits.

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
_ Age bot world: https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha
_ GTEA+ Gaze: https://cbs.ic.gatech.edu/fpv/
_ YouCook2 dataset
_ HOWTO100M: https://www.di.ens.fr/willow/research/howto100m/

### Implemented Loss (Spatial-Aware Composite Loss)

Our implementation uses a **composite loss with spatial awareness** to address the limitations of standard contrastive learning (e.g., CLIP's inability to distinguish "move left" vs "move right"). The loss has three components:

##### 1) Progress Regression Loss (L_prog)

Predicts normalized progress values for each timestep:

$$
L_{\text{prog}} = \text{MSE}(\sigma(z(V_t)), y_t)
$$

where $z(·)$ is z-score normalization, $\sigma$ is sigmoid, and $y_t \in [0,1]$ is the progress label.
**Purpose:** Grounds the model in actual task progress, not just visual-language similarity.

##### 2) Spatial-Aware InfoNCE Loss (L_spatial_nce)

Instead of using pooled features, we:

- Extract spatial patch features from SigLIP2's last_hidden_state (e.g., 16×16 patches)
- Use cross-attention where language queries attend to relevant spatial regions
- Compute contrastive loss on the attended spatial features

$$
L_{\text{spatial-nce}} = -\log \frac{\exp(s_{ii}/\tau)}{\sum_j \exp(s_{ij}/\tau)}
$$

where $s_{ij}$ is the similarity between spatially-attended features from trajectory $i$ and language $j$.
**Purpose:** Preserves spatial information that pooling discards, enabling distinction of spatial relationships.

##### 3) ReWiND Reversible Ranking Loss (L_rewind)

Based on ReWiND's key insight: learn from both forward AND reversed trajectories.
The loss has two components:

- **Forward ranking**: Sample (far, near) pairs where near is later in time, enforce $V_{\text{near}} > V_{\text{far}}$
- **Reverse ranking**: Reverse the trajectory and invert progress labels, then apply same ranking

$$
L_{\text{rewind}} = L_{\text{forward}} + L_{\text{reverse}}
$$

where both use: $\text{softplus}(m - (V_{\text{near}} - V_{\text{far}}))$

**Purpose:** By training on reversed trajectories with inverted progress, the model learns to distinguish progress from undoing progress. This is ReWiND's core contribution - understanding that tasks can be reversible.

##### Total Loss:

$$
L = \lambda_{\text{prog}} L_{\text{prog}} + \lambda_{\text{spatial-nce}} L_{\text{spatial-nce}} + \lambda_{\text{rewind}} L_{\text{rewind}}
$$

Default weights: $\lambda_{\text{prog}}=1.0$, $\lambda_{\text{spatial-nce}}=0.5$, $\lambda_{\text{rewind}}=0.4$

### TODOs:

- Implement first architecture [x]
- Implement processors [x]
- Choose right loss metric(s) [x]
- Make dataset with script that generated the dataset (IPEC-COMMUNITY/bc_z_lerobot) ready in lerobot format (and be able to visualize in dataset visualizer)
  - Annotate with ReWiND-style 0→1 progress rewards [x]
  - Visualize to check [x]
- Implement eval score or metric that is robust and can deal with generalization/is a good metric to try different architectures. And use it in an eval jupyter notebook with visalization of the live reward next to the video for part of the dataset: VOC score and score with correct and incorrect language captions [x]
- Do first training [x]
- Try different losses []
  - Only vlc loss then eval []
  - Only rewind loss then eval []
  - Vlc + rewind loss then eval []
- Cleanup code
- Switch to DINO v3 as encoder Base 86 M: https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m with HuggingFaceTB/SmolLM2-135M-Instruct ?
- Add more artificial text to dataset generated by vlm (google gemini) []
  - See google gemini vlm caption from Leandro [] https://gemini.google.com/app/7e332ffaf32580f2
  - Multiple captions per video, creat method to generate as much data as possible etc [] https://arxiv.org/abs/2508.13446
- How can we improve spatial aware learning? co generating captions for each frame with language decoder?
- Add droid []
- Extend evaluation []
- Add other dataset mentioned above []
