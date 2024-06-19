# Using `lerobot`  on a real world arm


In this example, we'll be using `lerobot` on a real world arm to:
- record a dataset in the `lerobot` format
- (soon) train a policy on it
- (soon) run the policy in the real-world

## Which robotic arm to use

In this example we're using the [open-source low-cost arm from Alexander Koch](https://github.com/AlexanderKoch-Koch/low_cost_robot) in the specific setup of:
- having 6 servos per arm, i.e. using the elbow-to-wrist extension
- adding two cameras around it, one on top and one in the front
- having a teleoperation arm as well (build the leader and the follower arms in A. Koch repo, both with elbow-to-wrist extensions)

I'm using these cameras (but the setup should not be sensitive to the exact cameras you're using):
- C922 Pro Stream Webcam
- Intel(R) RealSense D455 (using only the RGB input)


In general, this example should be very easily extendable to any type of arm using Dynamixel servos with at least one camera by changing a couple of configuration in the gym env.

## Install the example

Follow these steps:
- install `lerobot`
- install the Dynamixel-sdk: ` pip install dynamixel-sdk`

## Usage

### 0 - record examples

Run the `record_training_data.py` example, selecting the duration and number of episodes you want to record, e.g.
```
DATA_DIR='./data' python record_training_data.py \
--repo-id=thomwolf/blue_red_sort \
--num-episodes=50 \
--num-frames=400 \
--gym-config=./train_config/env/gym_real_world.yaml
```

TODO:
- various length episodes
- being able to drop episodes
- checking uploading to the hub

### 1 - visualize the dataset

Use the standard dataset visualization script pointing it to the right folder:
```
DATA_DIR='./data' python ../../lerobot/scripts/visualize_dataset.py \
    --repo-id thomwolf/blue_red_sort \
    --episode-index 0
```

### 2 - Train a policy

From the example directory let's run this command to train a model using ACT

```
DATA_DIR='./data' python ../../lerobot/scripts/train.py \
    device=cuda \
    hydra.searchpath=[file://./train_config/] \
    hydra.run.dir=./outputs/train/blue_red_sort \
    dataset_repo_id=thomwolf/blue_red_sort \
    env=gym_real_world \
    policy=act_real_world \
    wandb.enable=false
```

### 3 - Evaluate the policy in the real world

From the example directory let's run this command to evaluate our policy.
The configuration for running the policy is in the checkpoint of the model.
You can override parameters as follow:

```
python run_policy.py \
    -p ./outputs/train/blue_red_sort/checkpoints/last/pretrained_model/
    env.episode_length=1000
```


## Convert a hdf5 dataset recorded with the original ACT repo

You can convert a dataset from the raw data format of HDF5 files like in: https://github.com/tonyzhaozh/act with the following command:

```
python ./lerobot/scripts/push_dataset_to_hub.py
```
