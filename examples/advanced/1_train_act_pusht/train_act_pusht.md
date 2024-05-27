In this tutorial we will learn how to adapt a policy configuration to be compatible with a new environment and dataset. As a concrete example, we will adapt the default configuration for ACT to be compatible with the PushT environment and dataset.

If you haven't already read our tutorial on the [training script and configuration tooling](../4_train_policy_with_script.md) please do so prior to tackling this tutorial.

Let's get started!

Suppose we want to train ACT for PushT. Well, there are aspects of the ACT configuration that are specific to the ALOHA environments, and these happen to be incompatible with PushT. Therefore, trying to run the following will almost certainly raise an exception of sorts (eg: feature dimension mismatch):

```bash
python lerobot/scripts/train.py policy=act env=pusht dataset_repo_id=lerobot/pusht
```

We need to adapt the parameters of the ACT policy configuration to the PushT environment. The most important ones are the image keys.

ALOHA's datasets and environments typically use a variable number of cameras. In `lerobot/configs/policy/act.yaml` you may notice two relevant sections. Here we show you the minimal diff needed to adjust to PushT:

```diff
override_dataset_stats:
-  observation.images.top:
+  observation.image:
    # stats from imagenet, since we use a pretrained vision model
    mean: [[[0.485]], [[0.456]], [[0.406]]]  # (c,1,1)
    std: [[[0.229]], [[0.224]], [[0.225]]]  # (c,1,1)

policy:
  input_shapes:
-    observation.images.top: [3, 480, 640]
+    observation.image: [3, 96, 96]
    observation.state: ["${env.state_dim}"]
  output_shapes:
    action: ["${env.action_dim}"]

  input_normalization_modes:
-    observation.images.top: mean_std
+    observation.image: mean_std
     observation.state: min_max
  output_normalization_modes:
    action: min_max
```

Here we've accounted for the following:
- PushT uses "observation.image" for its image key.
- PushT provides smaller images.

_Side note: technically we could override these via the CLI, but with many changes it gets a bit messy, and we also have a bit of a challenge in that we're using `.` in our observation keys which is treated by Hydra as a hierarchical separator_.

For your convenience, we provide [`act_pusht.yaml`](./act_pusht.yaml) in this directory. It contains the diff above, plus some other (optional) ones that are explained within. Please copy it into `lerobot/configs/policy` with:

```bash
cp examples/advanced/1_train_act_pusht/act_pusht.yaml lerobot/configs/policy/act_pusht.yaml
```

(remember from a [previous tutorial](../4_train_policy_with_script.md) that Hydra will look in the `lerobot/configs` directory). Now try running the following.

<!-- Note to contributor: are you changing this command? Note that it's tested in `Makefile`, so change it there too! -->
```bash
python lerobot/scripts/train.py policy=act_pusht env=pusht
```

Notice that this is much the same as the command that failed at the start of the tutorial, only:
- Now we are using `policy=act_pusht` to point to our new configuration file.
- We can drop `dataset_repo_id=lerobot/pusht` as the change is incorporated in our new configuration file.

Hurrah! You're now training ACT for the PushT environment.

---

The bottom line of this tutorial is that when training policies for different environments and datasets you will need to understand what parts of the policy configuration are specific to those and make changes accordingly.

Happy coding! 🤗
