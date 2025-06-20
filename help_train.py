usage: train.py [-h] [--config_path str] [--dataset str]
                [--dataset.repo_id str|List] [--dataset.root [str]]
                [--dataset.episodes [List]] [--image_transforms str]
                [--dataset.image_transforms.enable bool]
                [--dataset.image_transforms.max_num_transforms int]
                [--dataset.image_transforms.random_order bool]
                [--dataset.image_transforms.tfs Dict]
                [--dataset.revision [str]] [--dataset.use_imagenet_stats bool]
                [--dataset.video_backend str] [--env str] [--env.obs_type str]
                [--env.render_mode str] [--env.visualization_width int]
                [--env.visualization_height int] [--robot str]
                [--env.robot.type {}] [--teleop str] [--env.teleop.type {}]
                [--env.task str] [--env.fps int] [--env.features Dict]
                [--env.features_map Dict] [--env.type str] [--env.name str]
                [--env.use_viewer bool] [--env.gripper_penalty float]
                [--env.use_gamepad bool] [--env.state_dim int]
                [--env.action_dim int] [--env.episode_length int]
                [--video_record str] [--env.video_record.enabled bool]
                [--env.video_record.record_dir str]
                [--env.video_record.trajectory_name str]
                [--env.reward_classifier_pretrained_path [str]]
                [--robot_config str] [--env.robot_config.type {}]
                [--teleop_config str] [--env.teleop_config.type {}]
                [--wrapper str] [--env.wrapper.control_mode str]
                [--env.wrapper.display_cameras bool]
                [--env.wrapper.add_joint_velocity_to_observation bool]
                [--env.wrapper.add_current_to_observation bool]
                [--env.wrapper.add_ee_pose_to_observation bool]
                [--env.wrapper.crop_params_dict [Dict]]
                [--env.wrapper.resize_size [int int]]
                [--env.wrapper.control_time_s float]
                [--env.wrapper.fixed_reset_joint_positions [Any]]
                [--env.wrapper.reset_time_s float]
                [--env.wrapper.use_gripper bool]
                [--env.wrapper.gripper_quantization_threshold [float]]
                [--env.wrapper.gripper_penalty float]
                [--env.wrapper.gripper_penalty_in_reward bool]
                [--env.mode str] [--env.repo_id [str]]
                [--env.dataset_root [str]] [--env.num_episodes int]
                [--env.episode int] [--env.device str]
                [--env.push_to_hub bool]
                [--env.pretrained_policy_name_or_path [str]]
                [--env.number_of_steps_after_success int] [--policy str]
                [--policy.type {act,diffusion,pi0,smolvla,tdmpc,vqbet,pi0fast,sac,reward_classifier}]
                [--policy.replace_final_stride_with_dilation int]
                [--policy.pre_norm bool] [--policy.dim_model int]
                [--policy.n_heads int] [--policy.dim_feedforward int]
                [--policy.feedforward_activation str]
                [--policy.n_encoder_layers int]
                [--policy.n_decoder_layers int] [--policy.use_vae bool]
                [--policy.n_vae_encoder_layers int]
                [--policy.temporal_ensemble_coeff [float]]
                [--policy.kl_weight float]
                [--policy.optimizer_lr_backbone float]
                [--policy.drop_n_last_frames int]
                [--policy.use_separate_rgb_encoder_per_camera bool]
                [--policy.down_dims int [int, ...]] [--policy.kernel_size int]
                [--policy.n_groups int]
                [--policy.diffusion_step_embed_dim int]
                [--policy.use_film_scale_modulation bool]
                [--policy.noise_scheduler_type str]
                [--policy.num_train_timesteps int]
                [--policy.beta_schedule str] [--policy.beta_start float]
                [--policy.beta_end float] [--policy.prediction_type str]
                [--policy.clip_sample bool] [--policy.clip_sample_range float]
                [--policy.num_inference_steps [int]]
                [--policy.do_mask_loss_for_padding bool]
                [--policy.scheduler_name str]
                [--policy.attention_implementation str]
                [--policy.num_steps int] [--policy.train_expert_only bool]
                [--policy.train_state_proj bool]
                [--policy.optimizer_grad_clip_norm float]
                [--policy.vlm_model_name str] [--policy.load_vlm_weights bool]
                [--policy.add_image_special_tokens bool]
                [--policy.attention_mode str] [--policy.prefix_length int]
                [--policy.pad_language_to str]
                [--policy.num_expert_layers int] [--policy.num_vlm_layers int]
                [--policy.self_attn_every_n_layers int]
                [--policy.expert_width_multiplier float]
                [--policy.min_period float] [--policy.max_period float]
                [--policy.n_action_repeats int] [--policy.horizon int]
                [--policy.q_ensemble_size int] [--policy.mlp_dim int]
                [--policy.use_mpc bool] [--policy.cem_iterations int]
                [--policy.max_std float] [--policy.min_std float]
                [--policy.n_gaussian_samples int] [--policy.n_pi_samples int]
                [--policy.uncertainty_regularizer_coeff float]
                [--policy.n_elites int]
                [--policy.elite_weighting_temperature float]
                [--policy.gaussian_mean_momentum float]
                [--policy.max_random_shift_ratio float]
                [--policy.reward_coeff float]
                [--policy.expectile_weight float] [--policy.value_coeff float]
                [--policy.consistency_coeff float]
                [--policy.advantage_scaling float] [--policy.pi_coeff float]
                [--policy.temporal_decay_coeff float]
                [--policy.target_model_momentum float]
                [--policy.n_action_pred_token int]
                [--policy.action_chunk_size int]
                [--policy.vision_backbone str] [--policy.crop_shape [int int]]
                [--policy.crop_is_random bool]
                [--policy.pretrained_backbone_weights [str]]
                [--policy.use_group_norm bool]
                [--policy.spatial_softmax_num_keypoints int]
                [--policy.n_vqvae_training_steps int]
                [--policy.vqvae_n_embed int]
                [--policy.vqvae_embedding_dim int]
                [--policy.vqvae_enc_hidden_dim int]
                [--policy.gpt_block_size int] [--policy.gpt_input_dim int]
                [--policy.gpt_output_dim int] [--policy.gpt_n_layer int]
                [--policy.gpt_n_head int] [--policy.gpt_hidden_dim int]
                [--policy.dropout float] [--policy.mlp_hidden_dim int]
                [--policy.offset_loss_weight float]
                [--policy.primary_code_loss_weight float]
                [--policy.secondary_code_loss_weight float]
                [--policy.bet_softmax_temperature float]
                [--policy.sequentially_select bool]
                [--policy.optimizer_vqvae_lr float]
                [--policy.optimizer_vqvae_weight_decay float]
                [--policy.chunk_size int] [--policy.n_action_steps int]
                [--policy.max_state_dim int] [--policy.max_action_dim int]
                [--policy.resize_imgs_with_padding int int]
                [--policy.interpolate_like_pi bool]
                [--policy.empty_cameras int] [--policy.adapt_to_pi_aloha bool]
                [--policy.use_delta_joint_actions_aloha bool]
                [--policy.tokenizer_max_length int] [--policy.proj_width int]
                [--policy.max_decoding_steps int]
                [--policy.fast_skip_tokens int]
                [--policy.max_input_seq_len int] [--policy.use_cache bool]
                [--policy.freeze_lm_head bool] [--policy.optimizer_lr float]
                [--policy.optimizer_betas float float]
                [--policy.optimizer_eps float]
                [--policy.optimizer_weight_decay float]
                [--policy.scheduler_warmup_steps int]
                [--policy.scheduler_decay_steps int]
                [--policy.scheduler_decay_lr float]
                [--policy.checkpoint_path str] [--policy.padding_side str]
                [--policy.precision str]
                [--policy.relaxed_action_decoding bool]
                [--policy.dataset_stats [Dict]] [--policy.storage_device str]
                [--policy.vision_encoder_name [str]]
                [--policy.freeze_vision_encoder bool]
                [--policy.image_encoder_hidden_dim int]
                [--policy.shared_encoder bool]
                [--policy.num_discrete_actions [int]]
                [--policy.online_steps int] [--policy.online_env_seed int]
                [--policy.online_buffer_capacity int]
                [--policy.offline_buffer_capacity int]
                [--policy.async_prefetch bool]
                [--policy.online_step_before_learning int]
                [--policy.policy_update_freq int] [--policy.discount float]
                [--policy.temperature_init float] [--policy.num_critics int]
                [--policy.num_subsample_critics [int]]
                [--policy.critic_lr float] [--policy.actor_lr float]
                [--policy.temperature_lr float]
                [--policy.critic_target_update_weight float]
                [--policy.utd_ratio int]
                [--policy.state_encoder_hidden_dim int]
                [--policy.target_entropy [float]]
                [--policy.use_backup_entropy bool]
                [--critic_network_kwargs str]
                [--policy.critic_network_kwargs.hidden_dims List]
                [--policy.critic_network_kwargs.activate_final bool]
                [--policy.critic_network_kwargs.final_activation [str]]
                [--actor_network_kwargs str]
                [--policy.actor_network_kwargs.hidden_dims List]
                [--policy.actor_network_kwargs.activate_final bool]
                [--policy_kwargs str]
                [--policy.policy_kwargs.use_tanh_squash bool]
                [--policy.policy_kwargs.std_min float]
                [--policy.policy_kwargs.std_max float]
                [--policy.policy_kwargs.init_final float]
                [--discrete_critic_network_kwargs str]
                [--policy.discrete_critic_network_kwargs.hidden_dims List]
                [--policy.discrete_critic_network_kwargs.activate_final bool]
                [--policy.discrete_critic_network_kwargs.final_activation [str]]
                [--actor_learner_config str]
                [--policy.actor_learner_config.learner_host str]
                [--policy.actor_learner_config.learner_port int]
                [--policy.actor_learner_config.policy_parameters_push_frequency int]
                [--policy.actor_learner_config.queue_get_timeout float]
                [--concurrency str] [--policy.concurrency.actor str]
                [--policy.concurrency.learner str]
                [--policy.use_torch_compile bool] [--policy.n_obs_steps int]
                [--policy.normalization_mapping Dict]
                [--policy.input_features Dict] [--policy.output_features Dict]
                [--policy.device str] [--policy.use_amp bool]
                [--policy.name str] [--policy.num_classes int]
                [--policy.hidden_dim int] [--policy.latent_dim int]
                [--policy.image_embedding_pooling_dim int]
                [--policy.dropout_rate float] [--policy.model_name str]
                [--policy.model_type str] [--policy.num_cameras int]
                [--policy.learning_rate float] [--policy.weight_decay float]
                [--policy.grad_clip_norm float] [--output_dir [Path]]
                [--job_name [str]] [--resume bool] [--seed [int]]
                [--num_workers int] [--batch_size int] [--steps int]
                [--eval_freq int] [--log_freq int] [--save_checkpoint bool]
                [--save_freq int] [--use_policy_training_preset bool]
                [--optimizer str]
                [--optimizer.type {adam,adamw,sgd,multi_adam}]
                [--optimizer.betas float float] [--optimizer.eps float]
                [--optimizer.momentum float] [--optimizer.dampening float]
                [--optimizer.nesterov bool] [--optimizer.lr float]
                [--optimizer.weight_decay float]
                [--optimizer.grad_clip_norm float]
                [--optimizer.optimizer_groups Dict] [--scheduler str]
                [--scheduler.type {diffuser,vqbet,cosine_decay_with_warmup}]
                [--scheduler.name str]
                [--scheduler.num_vqvae_training_steps int]
                [--scheduler.num_cycles float]
                [--scheduler.num_warmup_steps int]
                [--scheduler.num_decay_steps int] [--scheduler.peak_lr float]
                [--scheduler.decay_lr float] [--eval str]
                [--eval.n_episodes int] [--eval.batch_size int]
                [--eval.use_async_envs bool] [--wandb str]
                [--wandb.enable bool] [--wandb.disable_artifact bool]
                [--wandb.project str] [--wandb.entity [str]]
                [--wandb.notes [str]] [--wandb.run_id [str]]
                [--wandb.mode [str]]

options:
  -h, --help            show this help message and exit
  --config_path str     Path for a config file to parse with draccus (default:
                        None)
  --dataset str         Config file for dataset (default: None)
  --image_transforms str
                        Config file for image_transforms (default: None)
  --env str             Config file for env (default: None)
  --robot str           Config file for robot (default: None)
  --teleop str          Config file for teleop (default: None)
  --wrapper str         Config file for wrapper (default: None)
  --video_record str    Config file for video_record (default: None)
  --robot_config str    Config file for robot_config (default: None)
  --teleop_config str   Config file for teleop_config (default: None)
  --wrapper str         Config file for wrapper (default: None)
  --policy str          Config file for policy (default: None)
  --critic_network_kwargs str
                        Config file for critic_network_kwargs (default: None)
  --actor_network_kwargs str
                        Config file for actor_network_kwargs (default: None)
  --policy_kwargs str   Config file for policy_kwargs (default: None)
  --discrete_critic_network_kwargs str
                        Config file for discrete_critic_network_kwargs
                        (default: None)
  --actor_learner_config str
                        Config file for actor_learner_config (default: None)
  --concurrency str     Config file for concurrency (default: None)
  --optimizer str       Config file for optimizer (default: None)
  --scheduler str       Config file for scheduler (default: None)
  --eval str            Config file for eval (default: None)
  --wandb str           Config file for wandb (default: None)

TrainPipelineConfig:

  --output_dir [Path]   Set `dir` to where you would like to save all of the
                        run outputs. If you run another training session with
                        the same value for `dir` its contents will be
                        overwritten unless you set `resume` to true. (default:
                        None)
  --job_name [str]
  --resume bool         Set `resume` to true to resume a previous run. In
                        order for this to work, you will need to make sure
                        `dir` is the directory of an existing run with at
                        least one checkpoint in it. Note that when resuming a
                        run, the default behavior is to use the configuration
                        from the checkpoint, regardless of what's provided
                        with the training command at the time of resumption.
                        (default: False)
  --seed [int]          `seed` is used for training (eg: model initialization,
                        dataset shuffling) AND for the evaluation
                        environments. (default: 1000)
  --num_workers int     Number of workers for the dataloader. (default: 4)
  --batch_size int
  --steps int
  --eval_freq int
  --log_freq int
  --save_checkpoint bool
  --save_freq int       Checkpoint is saved every `save_freq` training
                        iterations and after the last training step. (default:
                        20000)
  --use_policy_training_preset bool

DatasetConfig ['dataset']:

  --dataset.repo_id str|List
                        You may provide a list of datasets here. `train.py`
                        creates them all and concatenates them. Note: only
                        data keys common between the datasets are kept. Each
                        dataset gets and additional transform that inserts the
                        "dataset_index" into the returned item. The index
                        mapping is made according to the order in which the
                        datasets are provided. (default: None)
  --dataset.root [str]  Root directory where the dataset will be stored (e.g.
                        'dataset/path'). (default: None)
  --dataset.episodes [List]
  --dataset.revision [str]
  --dataset.use_imagenet_stats bool
  --dataset.video_backend str

ImageTransformsConfig ['dataset.image_transforms']:

      These transforms are all using standard torchvision.transforms.v2
      You can find out how these transformations affect images here:
      https://pytorch.org/vision/0.18/auto_examples/transforms/plot_transforms_illustrations.html
      We use a custom RandomSubsetApply container to sample them.


  --dataset.image_transforms.enable bool
                        Set this flag to `true` to enable transforms during
                        training (default: False)
  --dataset.image_transforms.max_num_transforms int
                        This is the maximum number of transforms (sampled from
                        these below) that will be applied to each frame. It's
                        an integer in the interval [1,
                        number_of_available_transforms]. (default: 3)
  --dataset.image_transforms.random_order bool
                        By default, transforms are applied in Torchvision's
                        suggested order (shown below). Set this to True to
                        apply them in a random order. (default: False)
  --dataset.image_transforms.tfs Dict

Optional ['env']:

EnvConfig ['env']:

  --env.type {aloha,pusht,xarm,gym_manipulator,hil}
                        Which type of EnvConfig ['env'] to use (default: None)

AlohaEnv ['env']:

  --env.task str
  --env.fps int
  --env.features Dict
  --env.features_map Dict
  --env.episode_length int
  --env.obs_type str
  --env.render_mode str

PushtEnv ['env']:

  --env.task str
  --env.fps int
  --env.features Dict
  --env.features_map Dict
  --env.episode_length int
  --env.obs_type str
  --env.render_mode str
  --env.visualization_width int
  --env.visualization_height int

XarmEnv ['env']:

  --env.task str
  --env.fps int
  --env.features Dict
  --env.features_map Dict
  --env.episode_length int
  --env.obs_type str
  --env.render_mode str
  --env.visualization_width int
  --env.visualization_height int

HILSerlRobotEnvConfig ['env']:
  Configuration for the HILSerlRobotEnv environment.

  --env.task str
  --env.fps int
  --env.features Dict
  --env.features_map Dict
  --env.name str
  --env.mode str        Either "record", "replay", None (default: None)
  --env.repo_id [str]
  --env.dataset_root [str]
  --env.num_episodes int
                        only for record mode (default: 10)
  --env.episode int
  --env.device str
  --env.push_to_hub bool
  --env.pretrained_policy_name_or_path [str]
  --env.reward_classifier_pretrained_path [str]
  --env.number_of_steps_after_success int
                        For the reward classifier, to record more positive
                        examples after a success (default: 0)

Optional ['env.robot']:

RobotConfig ['env.robot']:

  --env.robot.type {}   Which type of RobotConfig ['env.robot'] to use
                        (default: None)

Optional ['env.teleop']:

TeleoperatorConfig ['env.teleop']:

  --env.teleop.type {}  Which type of TeleoperatorConfig ['env.teleop'] to use
                        (default: None)

Optional ['env.wrapper']:

EnvTransformConfig ['env.wrapper']:
  Configuration for environment wrappers.

  --env.wrapper.control_mode str
                        ee_action_space_params: EEActionSpaceConfig =
                        field(default_factory=EEActionSpaceConfig) (default:
                        gamepad)
  --env.wrapper.display_cameras bool
  --env.wrapper.add_joint_velocity_to_observation bool
  --env.wrapper.add_current_to_observation bool
  --env.wrapper.add_ee_pose_to_observation bool
  --env.wrapper.crop_params_dict [Dict]
  --env.wrapper.resize_size [int int]
  --env.wrapper.control_time_s float
  --env.wrapper.fixed_reset_joint_positions [Any]
  --env.wrapper.reset_time_s float
  --env.wrapper.use_gripper bool
  --env.wrapper.gripper_quantization_threshold [float]
  --env.wrapper.gripper_penalty float
  --env.wrapper.gripper_penalty_in_reward bool

HILEnvConfig ['env']:
  Configuration for the HIL environment.

  --env.task str
  --env.fps int
  --env.features Dict
  --env.features_map Dict
  --env.type str
  --env.name str
  --env.use_viewer bool
  --env.gripper_penalty float
  --env.use_gamepad bool
  --env.state_dim int
  --env.action_dim int
  --env.episode_length int
  --env.reward_classifier_pretrained_path [str]
                        ################ args from hilserlrobotenv (default:
                        None)
  --env.mode str        Either "record", "replay", None (default: None)
  --env.repo_id [str]
  --env.dataset_root [str]
  --env.num_episodes int
                        only for record mode (default: 10)
  --env.episode int
  --env.device str
  --env.push_to_hub bool
  --env.pretrained_policy_name_or_path [str]
  --env.number_of_steps_after_success int
                        For the reward classifier, to record more positive
                        examples after a success (default: 0)

VideoRecordConfig ['env.video_record']:
  Configuration for video recording in ManiSkill environments.

  --env.video_record.enabled bool
  --env.video_record.record_dir str
  --env.video_record.trajectory_name str

Optional ['env.robot_config']:

RobotConfig ['env.robot_config']:

  --env.robot_config.type {}
                        Which type of RobotConfig ['env.robot_config'] to use
                        (default: None)

Optional ['env.teleop_config']:

TeleoperatorConfig ['env.teleop_config']:

  --env.teleop_config.type {}
                        Which type of TeleoperatorConfig ['env.teleop_config']
                        to use (default: None)

Optional ['env.wrapper']:

EnvTransformConfig ['env.wrapper']:
  Configuration for environment wrappers.

  --env.wrapper.control_mode str
                        ee_action_space_params: EEActionSpaceConfig =
                        field(default_factory=EEActionSpaceConfig) (default:
                        gamepad)
  --env.wrapper.display_cameras bool
  --env.wrapper.add_joint_velocity_to_observation bool
  --env.wrapper.add_current_to_observation bool
  --env.wrapper.add_ee_pose_to_observation bool
  --env.wrapper.crop_params_dict [Dict]
  --env.wrapper.resize_size [int int]
  --env.wrapper.control_time_s float
  --env.wrapper.fixed_reset_joint_positions [Any]
  --env.wrapper.reset_time_s float
  --env.wrapper.use_gripper bool
  --env.wrapper.gripper_quantization_threshold [float]
  --env.wrapper.gripper_penalty float
  --env.wrapper.gripper_penalty_in_reward bool

Optional ['policy']:

PreTrainedConfig ['policy']:

      Base configuration class for policy models.

      Args:
          n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
              current step and additional steps going back).
          input_shapes: A dictionary defining the shapes of the input data for the policy.
          output_shapes: A dictionary defining the shapes of the output data for the policy.
          input_normalization_modes: A dictionary with key representing the modality and the value specifies the
              normalization mode to apply.
          output_normalization_modes: Similar dictionary as `input_normalization_modes`, but to unnormalize to
              the original scale.


  --policy.type {act,diffusion,pi0,smolvla,tdmpc,vqbet,pi0fast,sac,reward_classifier}
                        Which type of PreTrainedConfig ['policy'] to use
                        (default: None)

ACTConfig ['policy']:
  Configuration class for the Action Chunking Transformers policy.

      Defaults are configured for training on bimanual Aloha tasks like "insertion" or "transfer".

      The parameters you will most likely need to change are the ones which depend on the environment / sensors.
      Those are: `input_shapes` and 'output_shapes`.

      Notes on the inputs and outputs:
          - Either:
              - At least one key starting with "observation.image is required as an input.
                AND/OR
              - The key "observation.environment_state" is required as input.
          - If there are multiple keys beginning with "observation.images." they are treated as multiple camera
            views. Right now we only support all images having the same shape.
          - May optionally work without an "observation.state" key for the proprioceptive robot state.
          - "action" is required as an output key.

      Args:
          n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
              current step and additional steps going back).
          chunk_size: The size of the action prediction "chunks" in units of environment steps.
          n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
              This should be no greater than the chunk size. For example, if the chunk size size 100, you may
              set this to 50. This would mean that the model predicts 100 steps worth of actions, runs 50 in the
              environment, and throws the other 50 out.
          input_shapes: A dictionary defining the shapes of the input data for the policy. The key represents
              the input data name, and the value is a list indicating the dimensions of the corresponding data.
              For example, "observation.image" refers to an input from a camera with dimensions [3, 96, 96],
              indicating it has three color channels and 96x96 resolution. Importantly, `input_shapes` doesn't
              include batch dimension or temporal dimension.
          output_shapes: A dictionary defining the shapes of the output data for the policy. The key represents
              the output data name, and the value is a list indicating the dimensions of the corresponding data.
              For example, "action" refers to an output shape of [14], indicating 14-dimensional actions.
              Importantly, `output_shapes` doesn't include batch dimension or temporal dimension.
          input_normalization_modes: A dictionary with key representing the modality (e.g. "observation.state"),
              and the value specifies the normalization mode to apply. The two available modes are "mean_std"
              which subtracts the mean and divides by the standard deviation and "min_max" which rescale in a
              [-1, 1] range.
          output_normalization_modes: Similar dictionary as `normalize_input_modes`, but to unnormalize to the
              original scale. Note that this is also used for normalizing the training targets.
          vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
          pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
              `None` means no pretrained weights.
          replace_final_stride_with_dilation: Whether to replace the ResNet's final 2x2 stride with a dilated
              convolution.
          pre_norm: Whether to use "pre-norm" in the transformer blocks.
          dim_model: The transformer blocks' main hidden dimension.
          n_heads: The number of heads to use in the transformer blocks' multi-head attention.
          dim_feedforward: The dimension to expand the transformer's hidden dimension to in the feed-forward
              layers.
          feedforward_activation: The activation to use in the transformer block's feed-forward layers.
          n_encoder_layers: The number of transformer layers to use for the transformer encoder.
          n_decoder_layers: The number of transformer layers to use for the transformer decoder.
          use_vae: Whether to use a variational objective during training. This introduces another transformer
              which is used as the VAE's encoder (not to be confused with the transformer encoder - see
              documentation in the policy class).
          latent_dim: The VAE's latent dimension.
          n_vae_encoder_layers: The number of transformer layers to use for the VAE's encoder.
          temporal_ensemble_coeff: Coefficient for the exponential weighting scheme to apply for temporal
              ensembling. Defaults to None which means temporal ensembling is not used. `n_action_steps` must be
              1 when using this feature, as inference needs to happen at every step to form an ensemble. For
              more information on how ensembling works, please see `ACTTemporalEnsembler`.
          dropout: Dropout to use in the transformer layers (see code for details).
          kl_weight: The weight to use for the KL-divergence component of the loss if the variational objective
              is enabled. Loss is then calculated as: `reconstruction_loss + kl_weight * kld_loss`.


  --policy.n_obs_steps int
  --policy.normalization_mapping Dict
  --policy.input_features Dict
  --policy.output_features Dict
  --policy.device [str]
                        cuda | cpu | mp (default: None)
  --policy.use_amp bool
                        `use_amp` determines whether to use Automatic Mixed
                        Precision (AMP) for training and evaluation. With AMP,
                        automatic gradient scaling is used. (default: False)
  --policy.chunk_size int
  --policy.n_action_steps int
  --policy.vision_backbone str
  --policy.pretrained_backbone_weights [str]
  --policy.replace_final_stride_with_dilation int
  --policy.pre_norm bool
  --policy.dim_model int
  --policy.n_heads int
  --policy.dim_feedforward int
  --policy.feedforward_activation str
  --policy.n_encoder_layers int
  --policy.n_decoder_layers int
  --policy.use_vae bool
  --policy.latent_dim int
  --policy.n_vae_encoder_layers int
  --policy.temporal_ensemble_coeff [float]
  --policy.dropout float
  --policy.kl_weight float
  --policy.optimizer_lr float
                        Training preset (default: 1e-05)
  --policy.optimizer_weight_decay float
  --policy.optimizer_lr_backbone float

DiffusionConfig ['policy']:
  Configuration class for DiffusionPolicy.

      Defaults are configured for training with PushT providing proprioceptive and single camera observations.

      The parameters you will most likely need to change are the ones which depend on the environment / sensors.
      Those are: `input_shapes` and `output_shapes`.

      Notes on the inputs and outputs:
          - "observation.state" is required as an input key.
          - Either:
              - At least one key starting with "observation.image is required as an input.
                AND/OR
              - The key "observation.environment_state" is required as input.
          - If there are multiple keys beginning with "observation.image" they are treated as multiple camera
            views. Right now we only support all images having the same shape.
          - "action" is required as an output key.

      Args:
          n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
              current step and additional steps going back).
          horizon: Diffusion model action prediction size as detailed in `DiffusionPolicy.select_action`.
          n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
              See `DiffusionPolicy.select_action` for more details.
          input_shapes: A dictionary defining the shapes of the input data for the policy. The key represents
              the input data name, and the value is a list indicating the dimensions of the corresponding data.
              For example, "observation.image" refers to an input from a camera with dimensions [3, 96, 96],
              indicating it has three color channels and 96x96 resolution. Importantly, `input_shapes` doesn't
              include batch dimension or temporal dimension.
          output_shapes: A dictionary defining the shapes of the output data for the policy. The key represents
              the output data name, and the value is a list indicating the dimensions of the corresponding data.
              For example, "action" refers to an output shape of [14], indicating 14-dimensional actions.
              Importantly, `output_shapes` doesn't include batch dimension or temporal dimension.
          input_normalization_modes: A dictionary with key representing the modality (e.g. "observation.state"),
              and the value specifies the normalization mode to apply. The two available modes are "mean_std"
              which subtracts the mean and divides by the standard deviation and "min_max" which rescale in a
              [-1, 1] range.
          output_normalization_modes: Similar dictionary as `normalize_input_modes`, but to unnormalize to the
              original scale. Note that this is also used for normalizing the training targets.
          vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
          crop_shape: (H, W) shape to crop images to as a preprocessing step for the vision backbone. Must fit
              within the image size. If None, no cropping is done.
          crop_is_random: Whether the crop should be random at training time (it's always a center crop in eval
              mode).
          pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
              `None` means no pretrained weights.
          use_group_norm: Whether to replace batch normalization with group normalization in the backbone.
              The group sizes are set to be about 16 (to be precise, feature_dim // 16).
          spatial_softmax_num_keypoints: Number of keypoints for SpatialSoftmax.
          use_separate_rgb_encoders_per_camera: Whether to use a separate RGB encoder for each camera view.
          down_dims: Feature dimension for each stage of temporal downsampling in the diffusion modeling Unet.
              You may provide a variable number of dimensions, therefore also controlling the degree of
              downsampling.
          kernel_size: The convolutional kernel size of the diffusion modeling Unet.
          n_groups: Number of groups used in the group norm of the Unet's convolutional blocks.
          diffusion_step_embed_dim: The Unet is conditioned on the diffusion timestep via a small non-linear
              network. This is the output dimension of that network, i.e., the embedding dimension.
          use_film_scale_modulation: FiLM (https://huggingface.co/papers/1709.07871) is used for the Unet conditioning.
              Bias modulation is used be default, while this parameter indicates whether to also use scale
              modulation.
          noise_scheduler_type: Name of the noise scheduler to use. Supported options: ["DDPM", "DDIM"].
          num_train_timesteps: Number of diffusion steps for the forward diffusion schedule.
          beta_schedule: Name of the diffusion beta schedule as per DDPMScheduler from Hugging Face diffusers.
          beta_start: Beta value for the first forward-diffusion step.
          beta_end: Beta value for the last forward-diffusion step.
          prediction_type: The type of prediction that the diffusion modeling Unet makes. Choose from "epsilon"
              or "sample". These have equivalent outcomes from a latent variable modeling perspective, but
              "epsilon" has been shown to work better in many deep neural network settings.
          clip_sample: Whether to clip the sample to [-`clip_sample_range`, +`clip_sample_range`] for each
              denoising step at inference time. WARNING: you will need to make sure your action-space is
              normalized to fit within this range.
          clip_sample_range: The magnitude of the clipping range as described above.
          num_inference_steps: Number of reverse diffusion steps to use at inference time (steps are evenly
              spaced). If not provided, this defaults to be the same as `num_train_timesteps`.
          do_mask_loss_for_padding: Whether to mask the loss when there are copy-padded actions. See
              `LeRobotDataset` and `load_previous_and_future_frames` for more information. Note, this defaults
              to False as the original Diffusion Policy implementation does the same.


  --policy.n_obs_steps int
  --policy.normalization_mapping Dict
  --policy.input_features Dict
  --policy.output_features Dict
  --policy.device [str]
                        cuda | cpu | mp (default: None)
  --policy.use_amp bool
                        `use_amp` determines whether to use Automatic Mixed
                        Precision (AMP) for training and evaluation. With AMP,
                        automatic gradient scaling is used. (default: False)
  --policy.horizon int
  --policy.n_action_steps int
  --policy.drop_n_last_frames int
                        horizon - n_action_steps - n_obs_steps + 1 (default:
                        7)
  --policy.vision_backbone str
  --policy.crop_shape [int int]
  --policy.crop_is_random bool
  --policy.pretrained_backbone_weights [str]
  --policy.use_group_norm bool
  --policy.spatial_softmax_num_keypoints int
  --policy.use_separate_rgb_encoder_per_camera bool
  --policy.down_dims int [int, ...]
  --policy.kernel_size int
  --policy.n_groups int
  --policy.diffusion_step_embed_dim int
  --policy.use_film_scale_modulation bool
  --policy.noise_scheduler_type str
                        Noise scheduler. (default: DDPM)
  --policy.num_train_timesteps int
  --policy.beta_schedule str
  --policy.beta_start float
  --policy.beta_end float
  --policy.prediction_type str
  --policy.clip_sample bool
  --policy.clip_sample_range float
  --policy.num_inference_steps [int]
  --policy.do_mask_loss_for_padding bool
  --policy.optimizer_lr float
                        Training presets (default: 0.0001)
  --policy.optimizer_betas Any
  --policy.optimizer_eps float
  --policy.optimizer_weight_decay float
  --policy.scheduler_name str
  --policy.scheduler_warmup_steps int

PI0Config ['policy']:

  --policy.n_obs_steps int
                        Input / output structure. (default: 1)
  --policy.normalization_mapping Dict
  --policy.input_features Dict
  --policy.output_features Dict
  --policy.device [str]
                        cuda | cpu | mp (default: None)
  --policy.use_amp bool
                        `use_amp` determines whether to use Automatic Mixed
                        Precision (AMP) for training and evaluation. With AMP,
                        automatic gradient scaling is used. (default: False)
  --policy.chunk_size int
  --policy.n_action_steps int
  --policy.max_state_dim int
                        Shorter state and action vectors will be padded
                        (default: 32)
  --policy.max_action_dim int
  --policy.resize_imgs_with_padding int int
                        Image preprocessing (default: (224, 224))
  --policy.empty_cameras int
                        Add empty images. Used by pi0_aloha_sim which adds the
                        empty left and right wrist cameras in addition to the
                        top camera. (default: 0)
  --policy.adapt_to_pi_aloha bool
                        Converts the joint and gripper values from the
                        standard Aloha space to the space used by the pi
                        internal runtime which was used to train the base
                        model. (default: False)
  --policy.use_delta_joint_actions_aloha bool
                        Converts joint dimensions to deltas with respect to
                        the current state before passing to the model. Gripper
                        dimensions will remain in absolute values. (default:
                        False)
  --policy.tokenizer_max_length int
                        Tokenizer (default: 48)
  --policy.proj_width int
                        Projector (default: 1024)
  --policy.num_steps int
                        Decoding (default: 10)
  --policy.use_cache bool
                        Attention utils (default: True)
  --policy.attention_implementation str
                        or fa2, flex (default: eager)
  --policy.freeze_vision_encoder bool
                        Finetuning settings (default: True)
  --policy.train_expert_only bool
  --policy.train_state_proj bool
  --policy.optimizer_lr float
                        Training presets (default: 2.5e-05)
  --policy.optimizer_betas float float
  --policy.optimizer_eps float
  --policy.optimizer_weight_decay float
  --policy.scheduler_warmup_steps int
  --policy.scheduler_decay_steps int
  --policy.scheduler_decay_lr float

SmolVLAConfig ['policy']:

  --policy.n_obs_steps int
                        Input / output structure. (default: 1)
  --policy.normalization_mapping Dict
  --policy.input_features Dict
  --policy.output_features Dict
  --policy.device [str]
                        cuda | cpu | mp (default: None)
  --policy.use_amp bool
                        `use_amp` determines whether to use Automatic Mixed
                        Precision (AMP) for training and evaluation. With AMP,
                        automatic gradient scaling is used. (default: False)
  --policy.chunk_size int
  --policy.n_action_steps int
  --policy.max_state_dim int
                        Shorter state and action vectors will be padded
                        (default: 32)
  --policy.max_action_dim int
  --policy.resize_imgs_with_padding int int
                        Image preprocessing (default: (512, 512))
  --policy.empty_cameras int
                        Add empty images. Used by smolvla_aloha_sim which adds
                        the empty left and right wrist cameras in addition to
                        the top camera. (default: 0)
  --policy.adapt_to_pi_aloha bool
                        Converts the joint and gripper values from the
                        standard Aloha space to the space used by the pi
                        internal runtime which was used to train the base
                        model. (default: False)
  --policy.use_delta_joint_actions_aloha bool
                        Converts joint dimensions to deltas with respect to
                        the current state before passing to the model. Gripper
                        dimensions will remain in absolute values. (default:
                        False)
  --policy.tokenizer_max_length int
                        Tokenizer (default: 48)
  --policy.num_steps int
                        Decoding (default: 10)
  --policy.use_cache bool
                        Attention utils (default: True)
  --policy.freeze_vision_encoder bool
                        Finetuning settings (default: True)
  --policy.train_expert_only bool
  --policy.train_state_proj bool
  --policy.optimizer_lr float
                        Training presets (default: 0.0001)
  --policy.optimizer_betas float float
  --policy.optimizer_eps float
  --policy.optimizer_weight_decay float
  --policy.optimizer_grad_clip_norm float
  --policy.scheduler_warmup_steps int
  --policy.scheduler_decay_steps int
  --policy.scheduler_decay_lr float
  --policy.vlm_model_name str
                        Select the VLM backbone. (default:
                        HuggingFaceTB/SmolVLM2-500M-Video-Instruct)
  --policy.load_vlm_weights bool
                        Set to True in case of training the expert from
                        scratch. True when init from pretrained SmolVLA
                        weights (default: False)
  --policy.add_image_special_tokens bool
                        Whether to use special image tokens around image
                        features. (default: False)
  --policy.attention_mode str
  --policy.prefix_length int
  --policy.pad_language_to str
                        "max_length" (default: longest)
  --policy.num_expert_layers int
                        Less or equal to 0 is the default where the action
                        expert has the same number of layers of VLM. Otherwise
                        the expert have less layers. (default: -1)
  --policy.num_vlm_layers int
                        Number of layers used in the VLM (first num_vlm_layers
                        layers) (default: 16)
  --policy.self_attn_every_n_layers int
                        Interleave SA layers each self_attn_every_n_layers
                        (default: 2)
  --policy.expert_width_multiplier float
                        The action expert hidden size (wrt to the VLM)
                        (default: 0.75)
  --policy.min_period float
                        sensitivity range for the timestep used in sine-cosine
                        positional encoding (default: 0.004)
  --policy.max_period float

TDMPCConfig ['policy']:
  Configuration class for TDMPCPolicy.

      Defaults are configured for training with xarm_lift_medium_replay providing proprioceptive and single
      camera observations.

      The parameters you will most likely need to change are the ones which depend on the environment / sensors.
      Those are: `input_shapes`, `output_shapes`, and perhaps `max_random_shift_ratio`.

      Args:
          n_action_repeats: The number of times to repeat the action returned by the planning. (hint: Google
              action repeats in Q-learning or ask your favorite chatbot)
          horizon: Horizon for model predictive control.
          n_action_steps: Number of action steps to take from the plan given by model predictive control. This
              is an alternative to using action repeats. If this is set to more than 1, then we require
              `n_action_repeats == 1`, `use_mpc == True` and `n_action_steps <= horizon`. Note that this
              approach of using multiple steps from the plan is not in the original implementation.
          input_shapes: A dictionary defining the shapes of the input data for the policy. The key represents
              the input data name, and the value is a list indicating the dimensions of the corresponding data.
              For example, "observation.image" refers to an input from a camera with dimensions [3, 96, 96],
              indicating it has three color channels and 96x96 resolution. Importantly, `input_shapes` doesn't
              include batch dimension or temporal dimension.
          output_shapes: A dictionary defining the shapes of the output data for the policy. The key represents
              the output data name, and the value is a list indicating the dimensions of the corresponding data.
              For example, "action" refers to an output shape of [14], indicating 14-dimensional actions.
              Importantly, `output_shapes` doesn't include batch dimension or temporal dimension.
          input_normalization_modes: A dictionary with key representing the modality (e.g. "observation.state"),
              and the value specifies the normalization mode to apply. The two available modes are "mean_std"
              which subtracts the mean and divides by the standard deviation and "min_max" which rescale in a
              [-1, 1] range. Note that here this defaults to None meaning inputs are not normalized. This is to
              match the original implementation.
          output_normalization_modes: Similar dictionary as `normalize_input_modes`, but to unnormalize to the
              original scale. Note that this is also used for normalizing the training targets. NOTE: Clipping
              to [-1, +1] is used during MPPI/CEM. Therefore, it is recommended that you stick with "min_max"
              normalization mode here.
          image_encoder_hidden_dim: Number of channels for the convolutional layers used for image encoding.
          state_encoder_hidden_dim: Hidden dimension for MLP used for state vector encoding.
          latent_dim: Observation's latent embedding dimension.
          q_ensemble_size: Number of Q function estimators to use in an ensemble for uncertainty estimation.
          mlp_dim: Hidden dimension of MLPs used for modelling the dynamics encoder, reward function, policy
              (π), Q ensemble, and V.
          discount: Discount factor (γ) to use for the reinforcement learning formalism.
          use_mpc: Whether to use model predictive control. The alternative is to just sample the policy model
              (π) for each step.
          cem_iterations: Number of iterations for the MPPI/CEM loop in MPC.
          max_std: Maximum standard deviation for actions sampled from the gaussian PDF in CEM.
          min_std: Minimum standard deviation for noise applied to actions sampled from the policy model (π).
              Doubles up as the minimum standard deviation for actions sampled from the gaussian PDF in CEM.
          n_gaussian_samples: Number of samples to draw from the gaussian distribution every CEM iteration. Must
              be non-zero.
          n_pi_samples: Number of samples to draw from the policy / world model rollout every CEM iteration. Can
              be zero.
          uncertainty_regularizer_coeff: Coefficient for the uncertainty regularization used when estimating
              trajectory values (this is the λ coefficient in eqn 4 of FOWM).
          n_elites: The number of elite samples to use for updating the gaussian parameters every CEM iteration.
          elite_weighting_temperature: The temperature to use for softmax weighting (by trajectory value) of the
              elites, when updating the gaussian parameters for CEM.
          gaussian_mean_momentum: Momentum (α) used for EMA updates of the mean parameter μ of the gaussian
              parameters optimized in CEM. Updates are calculated as μ⁻ ← αμ⁻ + (1-α)μ.
          max_random_shift_ratio: Maximum random shift (as a proportion of the image size) to apply to the
              image(s) (in units of pixels) for training-time augmentation. If set to 0, no such augmentation
              is applied. Note that the input images are assumed to be square for this augmentation.
          reward_coeff: Loss weighting coefficient for the reward regression loss.
          expectile_weight: Weighting (τ) used in expectile regression for the state value function (V).
              v_pred < v_target is weighted by τ and v_pred >= v_target is weighted by (1-τ). τ is expected to
              be in [0, 1]. Setting τ closer to 1 results in a more "optimistic" V. This is sensible to do
              because v_target is obtained by evaluating the learned state-action value functions (Q) with
              in-sample actions that may not be always optimal.
          value_coeff: Loss weighting coefficient for both the state-action value (Q) TD loss, and the state
              value (V) expectile regression loss.
          consistency_coeff: Loss weighting coefficient for the consistency loss.
          advantage_scaling: A factor by which the advantages are scaled prior to exponentiation for advantage
              weighted regression of the policy (π) estimator parameters. Note that the exponentiated advantages
              are clamped at 100.0.
          pi_coeff: Loss weighting coefficient for the action regression loss.
          temporal_decay_coeff: Exponential decay coefficient for decaying the loss coefficient for future time-
              steps. Hint: each loss computation involves `horizon` steps worth of actions starting from the
              current time step.
          target_model_momentum: Momentum (α) used for EMA updates of the target models. Updates are calculated
              as ϕ ← αϕ + (1-α)θ where ϕ are the parameters of the target model and θ are the parameters of the
              model being trained.


  --policy.n_obs_steps int
                        Input / output structure. (default: 1)
  --policy.normalization_mapping Dict
  --policy.input_features Dict
  --policy.output_features Dict
  --policy.device [str]
                        cuda | cpu | mp (default: None)
  --policy.use_amp bool
                        `use_amp` determines whether to use Automatic Mixed
                        Precision (AMP) for training and evaluation. With AMP,
                        automatic gradient scaling is used. (default: False)
  --policy.n_action_repeats int
  --policy.horizon int
  --policy.n_action_steps int
  --policy.image_encoder_hidden_dim int
  --policy.state_encoder_hidden_dim int
  --policy.latent_dim int
  --policy.q_ensemble_size int
  --policy.mlp_dim int
  --policy.discount float
  --policy.use_mpc bool
  --policy.cem_iterations int
  --policy.max_std float
  --policy.min_std float
  --policy.n_gaussian_samples int
  --policy.n_pi_samples int
  --policy.uncertainty_regularizer_coeff float
  --policy.n_elites int
  --policy.elite_weighting_temperature float
  --policy.gaussian_mean_momentum float
  --policy.max_random_shift_ratio float
  --policy.reward_coeff float
  --policy.expectile_weight float
  --policy.value_coeff float
  --policy.consistency_coeff float
  --policy.advantage_scaling float
  --policy.pi_coeff float
  --policy.temporal_decay_coeff float
  --policy.target_model_momentum float
  --policy.optimizer_lr float
                        Training presets (default: 0.0003)

VQBeTConfig ['policy']:
  Configuration class for VQ-BeT.

      Defaults are configured for training with PushT providing proprioceptive and single camera observations.

      The parameters you will most likely need to change are the ones which depend on the environment / sensors.
      Those are: `input_shapes` and `output_shapes`.

      Notes on the inputs and outputs:
          - "observation.state" is required as an input key.
          - At least one key starting with "observation.image is required as an input.
          - If there are multiple keys beginning with "observation.image" they are treated as multiple camera
            views. Right now we only support all images having the same shape.
          - "action" is required as an output key.

      Args:
          n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
              current step and additional steps going back).
          n_action_pred_token: Total number of current token and future tokens that VQ-BeT predicts.
          action_chunk_size: Action chunk size of each action prediction token.
          input_shapes: A dictionary defining the shapes of the input data for the policy.
              The key represents the input data name, and the value is a list indicating the dimensions
              of the corresponding data. For example, "observation.image" refers to an input from
              a camera with dimensions [3, 96, 96], indicating it has three color channels and 96x96 resolution.
              Importantly, shapes doesnt include batch dimension or temporal dimension.
          output_shapes: A dictionary defining the shapes of the output data for the policy.
              The key represents the output data name, and the value is a list indicating the dimensions
              of the corresponding data. For example, "action" refers to an output shape of [14], indicating
              14-dimensional actions. Importantly, shapes doesnt include batch dimension or temporal dimension.
          input_normalization_modes: A dictionary with key representing the modality (e.g. "observation.state"),
              and the value specifies the normalization mode to apply. The two available modes are "mean_std"
              which subtracts the mean and divides by the standard deviation and "min_max" which rescale in a
              [-1, 1] range.
          output_normalization_modes: Similar dictionary as `normalize_input_modes`, but to unnormalize to the
              original scale. Note that this is also used for normalizing the training targets.
          vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
          crop_shape: (H, W) shape to crop images to as a preprocessing step for the vision backbone. Must fit
              within the image size. If None, no cropping is done.
          crop_is_random: Whether the crop should be random at training time (it's always a center crop in eval
              mode).
          pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
              `None` means no pretrained weights.
          use_group_norm: Whether to replace batch normalization with group normalization in the backbone.
              The group sizes are set to be about 16 (to be precise, feature_dim // 16).
          spatial_softmax_num_keypoints: Number of keypoints for SpatialSoftmax.
          n_vqvae_training_steps: Number of optimization steps for training Residual VQ.
          vqvae_n_embed: Number of embedding vectors in the RVQ dictionary (each layer).
          vqvae_embedding_dim: Dimension of each embedding vector in the RVQ dictionary.
          vqvae_enc_hidden_dim: Size of hidden dimensions of Encoder / Decoder part of Residaul VQ-VAE
          gpt_block_size: Max block size of minGPT (should be larger than the number of input tokens)
          gpt_input_dim: Size of output input of GPT. This is also used as the dimension of observation features.
          gpt_output_dim: Size of output dimension of GPT. This is also used as a input dimension of offset / bin prediction headers.
          gpt_n_layer: Number of layers of GPT
          gpt_n_head: Number of headers of GPT
          gpt_hidden_dim: Size of hidden dimensions of GPT
          dropout: Dropout rate for GPT
          mlp_hidden_dim: Size of hidden dimensions of offset header / bin prediction headers parts of VQ-BeT
          offset_loss_weight:  A constant that is multiplied to the offset loss
          primary_code_loss_weight: A constant that is multiplied to the primary code prediction loss
          secondary_code_loss_weight: A constant that is multiplied to the secondary code prediction loss
          bet_softmax_temperature: Sampling temperature of code for rollout with VQ-BeT
          sequentially_select: Whether select code of primary / secondary as sequentially (pick primary code,
              and then select secodnary code), or at the same time.


  --policy.n_obs_steps int
  --policy.normalization_mapping Dict
  --policy.input_features Dict
  --policy.output_features Dict
  --policy.device [str]
                        cuda | cpu | mp (default: None)
  --policy.use_amp bool
                        `use_amp` determines whether to use Automatic Mixed
                        Precision (AMP) for training and evaluation. With AMP,
                        automatic gradient scaling is used. (default: False)
  --policy.n_action_pred_token int
  --policy.action_chunk_size int
  --policy.vision_backbone str
  --policy.crop_shape [int int]
  --policy.crop_is_random bool
  --policy.pretrained_backbone_weights [str]
  --policy.use_group_norm bool
  --policy.spatial_softmax_num_keypoints int
  --policy.n_vqvae_training_steps int
  --policy.vqvae_n_embed int
  --policy.vqvae_embedding_dim int
  --policy.vqvae_enc_hidden_dim int
  --policy.gpt_block_size int
  --policy.gpt_input_dim int
  --policy.gpt_output_dim int
  --policy.gpt_n_layer int
  --policy.gpt_n_head int
  --policy.gpt_hidden_dim int
  --policy.dropout float
  --policy.mlp_hidden_dim int
  --policy.offset_loss_weight float
  --policy.primary_code_loss_weight float
  --policy.secondary_code_loss_weight float
  --policy.bet_softmax_temperature float
  --policy.sequentially_select bool
  --policy.optimizer_lr float
                        Training presets (default: 0.0001)
  --policy.optimizer_betas Any
  --policy.optimizer_eps float
  --policy.optimizer_weight_decay float
  --policy.optimizer_vqvae_lr float
  --policy.optimizer_vqvae_weight_decay float
  --policy.scheduler_warmup_steps int

PI0FASTConfig ['policy']:

  --policy.n_obs_steps int
                        Input / output structure. (default: 1)
  --policy.normalization_mapping Dict
  --policy.input_features Dict
  --policy.output_features Dict
  --policy.device [str]
                        cuda | cpu | mp (default: None)
  --policy.use_amp bool
                        `use_amp` determines whether to use Automatic Mixed
                        Precision (AMP) for training and evaluation. With AMP,
                        automatic gradient scaling is used. (default: False)
  --policy.chunk_size int
  --policy.n_action_steps int
  --policy.max_state_dim int
                        32 (default: 32)
  --policy.max_action_dim int
                        32 (default: 32)
  --policy.resize_imgs_with_padding int int
                        Image preprocessing (default: (224, 224))
  --policy.interpolate_like_pi bool
  --policy.empty_cameras int
                        Add empty images. Used by pi0_aloha_sim which adds the
                        empty left and right wrist cameras in addition to the
                        top camera. (default: 0)
  --policy.adapt_to_pi_aloha bool
                        Converts the joint and gripper values from the
                        standard Aloha space to the space used by the pi
                        internal runtime which was used to train the base
                        model. (default: False)
  --policy.use_delta_joint_actions_aloha bool
                        Converts joint dimensions to deltas with respect to
                        the current state before passing to the model. Gripper
                        dimensions will remain in absolute values. (default:
                        False)
  --policy.tokenizer_max_length int
                        Tokenizer (default: 48)
  --policy.proj_width int
                        Projector (default: 1024)
  --policy.max_decoding_steps int
                        Decoding (default: 256)
  --policy.fast_skip_tokens int
                        Skip last 128 tokens in PaliGemma vocab since they are
                        special tokens (default: 128)
  --policy.max_input_seq_len int
                        512 (default: 256)
  --policy.use_cache bool
                        Utils (default: True)
  --policy.freeze_vision_encoder bool
                        Frozen parameters (default: True)
  --policy.freeze_lm_head bool
  --policy.optimizer_lr float
                        Training presets (default: 0.0001)
  --policy.optimizer_betas float float
  --policy.optimizer_eps float
  --policy.optimizer_weight_decay float
  --policy.scheduler_warmup_steps int
  --policy.scheduler_decay_steps int
  --policy.scheduler_decay_lr float
  --policy.checkpoint_path str
  --policy.padding_side str
  --policy.precision str
  --policy.grad_clip_norm float
  --policy.relaxed_action_decoding bool
                        Allows padding/truncation of generated action tokens
                        during detokenization to ensure decoding. In the
                        original version, tensors of 0s were generated if
                        shapes didn't match for stable decoding. (default:
                        True)

SACConfig ['policy']:
  Soft Actor-Critic (SAC) configuration.

      SAC is an off-policy actor-critic deep RL algorithm based on the maximum entropy
      reinforcement learning framework. It learns a policy and a Q-function simultaneously
      using experience collected from the environment.

      This configuration class contains all the parameters needed to define a SAC agent,
      including network architectures, optimization settings, and algorithm-specific
      hyperparameters.


  --policy.n_obs_steps int
  --policy.normalization_mapping Dict
                        Mapping of feature types to normalization modes
                        (default: {'VISUAL': <NormalizationMode.MEAN_STD:
                        'MEAN_STD'>, 'STATE': <NormalizationMode.MIN_MAX:
                        'MIN_MAX'>, 'ENV': <NormalizationMode.MIN_MAX:
                        'MIN_MAX'>, 'ACTION': <NormalizationMode.MIN_MAX:
                        'MIN_MAX'>})
  --policy.input_features Dict
  --policy.output_features Dict
  --policy.device str   Architecture specifics Device to run the model on
                        (e.g., "cuda", "cpu") (default: cpu)
  --policy.use_amp bool
                        `use_amp` determines whether to use Automatic Mixed
                        Precision (AMP) for training and evaluation. With AMP,
                        automatic gradient scaling is used. (default: False)
  --policy.dataset_stats [Dict]
                        Statistics for normalizing different types of inputs
                        (default: {'observation.image': {'mean': [0.485,
                        0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
                        'observation.state': {'min': [0.0, 0.0], 'max': [1.0,
                        1.0]}, 'action': {'min': [0.0, 0.0, 0.0], 'max': [1.0,
                        1.0, 1.0]}})
  --policy.storage_device str
                        Device to store the model on (default: cpu)
  --policy.vision_encoder_name [str]
                        Name of the vision encoder model (Set to
                        "helper2424/resnet10" for hil serl resnet10) (default:
                        None)
  --policy.freeze_vision_encoder bool
                        Whether to freeze the vision encoder during training
                        (default: True)
  --policy.image_encoder_hidden_dim int
                        Hidden dimension size for the image encoder (default:
                        32)
  --policy.shared_encoder bool
                        Whether to use a shared encoder for actor and critic
                        (default: True)
  --policy.num_discrete_actions [int]
                        Number of discrete actions, eg for gripper actions
                        (default: None)
  --policy.image_embedding_pooling_dim int
                        Dimension of the image embedding pooling (default: 8)
  --policy.online_steps int
                        Training parameter Number of steps for online training
                        (default: 1000000)
  --policy.online_env_seed int
                        Seed for the online environment (default: 10000)
  --policy.online_buffer_capacity int
                        Capacity of the online replay buffer (default: 100000)
  --policy.offline_buffer_capacity int
                        Capacity of the offline replay buffer (default:
                        100000)
  --policy.async_prefetch bool
                        Whether to use asynchronous prefetching for the
                        buffers (default: False)
  --policy.online_step_before_learning int
                        Number of steps before learning starts (default: 100)
  --policy.policy_update_freq int
                        Frequency of policy updates (default: 1)
  --policy.discount float
                        SAC algorithm parameters Discount factor for the SAC
                        algorithm (default: 0.99)
  --policy.temperature_init float
                        Initial temperature value (default: 1.0)
  --policy.num_critics int
                        Number of critics in the ensemble (default: 2)
  --policy.num_subsample_critics [int]
                        Number of subsampled critics for training (default:
                        None)
  --policy.critic_lr float
                        Learning rate for the critic network (default: 0.0003)
  --policy.actor_lr float
                        Learning rate for the actor network (default: 0.0003)
  --policy.temperature_lr float
                        Learning rate for the temperature parameter (default:
                        0.0003)
  --policy.critic_target_update_weight float
                        Weight for the critic target update (default: 0.005)
  --policy.utd_ratio int
                        Update-to-data ratio for the UTD algorithm (If you
                        want enable utd_ratio, you need to set it to >1)
                        (default: 1)
  --policy.state_encoder_hidden_dim int
                        Hidden dimension size for the state encoder (default:
                        256)
  --policy.latent_dim int
                        Dimension of the latent space (default: 256)
  --policy.target_entropy [float]
                        Target entropy for the SAC algorithm (default: None)
  --policy.use_backup_entropy bool
                        Whether to use backup entropy for the SAC algorithm
                        (default: True)
  --policy.grad_clip_norm float
                        Gradient clipping norm for the SAC algorithm (default:
                        40.0)
  --policy.use_torch_compile bool
                        Optimizations (default: True)

CriticNetworkConfig ['policy.critic_network_kwargs']:
  Network configuration
  Configuration for the critic network architecture

  --policy.critic_network_kwargs.hidden_dims List
  --policy.critic_network_kwargs.activate_final bool
  --policy.critic_network_kwargs.final_activation [str]

ActorNetworkConfig ['policy.actor_network_kwargs']:
  Configuration for the actor network architecture

  --policy.actor_network_kwargs.hidden_dims List
  --policy.actor_network_kwargs.activate_final bool

PolicyConfig ['policy.policy_kwargs']:
  Configuration for the policy parameters

  --policy.policy_kwargs.use_tanh_squash bool
  --policy.policy_kwargs.std_min float
  --policy.policy_kwargs.std_max float
  --policy.policy_kwargs.init_final float

CriticNetworkConfig ['policy.discrete_critic_network_kwargs']:
  Configuration for the discrete critic network

  --policy.discrete_critic_network_kwargs.hidden_dims List
  --policy.discrete_critic_network_kwargs.activate_final bool
  --policy.discrete_critic_network_kwargs.final_activation [str]

ActorLearnerConfig ['policy.actor_learner_config']:
  Configuration for actor-learner architecture

  --policy.actor_learner_config.learner_host str
  --policy.actor_learner_config.learner_port int
  --policy.actor_learner_config.policy_parameters_push_frequency int
  --policy.actor_learner_config.queue_get_timeout float

ConcurrencyConfig ['policy.concurrency']:
  Configuration for concurrency settings (you can use threads or processes for the actor and learner)

  --policy.concurrency.actor str
  --policy.concurrency.learner str

RewardClassifierConfig ['policy']:
  Configuration for the Reward Classifier model.

  --policy.n_obs_steps int
  --policy.normalization_mapping Dict
  --policy.input_features Dict
  --policy.output_features Dict
  --policy.device str
  --policy.use_amp bool
                        `use_amp` determines whether to use Automatic Mixed
                        Precision (AMP) for training and evaluation. With AMP,
                        automatic gradient scaling is used. (default: False)
  --policy.name str
  --policy.num_classes int
  --policy.hidden_dim int
  --policy.latent_dim int
  --policy.image_embedding_pooling_dim int
  --policy.dropout_rate float
  --policy.model_name str
  --policy.model_type str
                        "transformer" or "cnn" (default: cnn)
  --policy.num_cameras int
  --policy.learning_rate float
  --policy.weight_decay float
  --policy.grad_clip_norm float

Optional ['optimizer']:

OptimizerConfig ['optimizer']:

  --optimizer.type {adam,adamw,sgd,multi_adam}
                        Which type of OptimizerConfig ['optimizer'] to use
                        (default: None)

AdamConfig ['optimizer']:

  --optimizer.lr float
  --optimizer.weight_decay float
  --optimizer.grad_clip_norm float
  --optimizer.betas float float
  --optimizer.eps float

AdamWConfig ['optimizer']:

  --optimizer.lr float
  --optimizer.weight_decay float
  --optimizer.grad_clip_norm float
  --optimizer.betas float float
  --optimizer.eps float

SGDConfig ['optimizer']:

  --optimizer.lr float
  --optimizer.weight_decay float
  --optimizer.grad_clip_norm float
  --optimizer.momentum float
  --optimizer.dampening float
  --optimizer.nesterov bool

MultiAdamConfig ['optimizer']:
  Configuration for multiple Adam optimizers with different parameter groups.

      This creates a dictionary of Adam optimizers, each with its own hyperparameters.

      Args:
          lr: Default learning rate (used if not specified for a group)
          weight_decay: Default weight decay (used if not specified for a group)
          optimizer_groups: Dictionary mapping parameter group names to their hyperparameters
          grad_clip_norm: Gradient clipping norm


  --optimizer.lr float
  --optimizer.weight_decay float
  --optimizer.grad_clip_norm float
                        lr: float = 1e-3 weight_decay: float = 0.0
                        grad_clip_norm: float = 10.0 optimizer_groups:
                        dict[str, dict[str, Any]] =
                        field(default_factory=dict) def build(self,
                        params_dict: dict[str, list]) -> dict[str,
                        torch.optim.Optimizer]: (default: 10.0)
  --optimizer.optimizer_groups Dict

Optional ['scheduler']:

LRSchedulerConfig ['scheduler']:

  --scheduler.type {diffuser,vqbet,cosine_decay_with_warmup}
                        Which type of LRSchedulerConfig ['scheduler'] to use
                        (default: None)

DiffuserSchedulerConfig ['scheduler']:

  --scheduler.num_warmup_steps [int]
  --scheduler.name str

VQBeTSchedulerConfig ['scheduler']:

  --scheduler.num_warmup_steps int
  --scheduler.num_vqvae_training_steps int
  --scheduler.num_cycles float

CosineDecayWithWarmupSchedulerConfig ['scheduler']:
  Used by Physical Intelligence to train Pi0

  --scheduler.num_warmup_steps int
  --scheduler.num_decay_steps int
  --scheduler.peak_lr float
  --scheduler.decay_lr float

EvalConfig ['eval']:

  --eval.n_episodes int
  --eval.batch_size int
                        `batch_size` specifies the number of environments to
                        use in a gym.vector.VectorEnv. (default: 50)
  --eval.use_async_envs bool
                        `use_async_envs` specifies whether to use asynchronous
                        environments (multiprocessing). (default: False)

WandBConfig ['wandb']:

  --wandb.enable bool
  --wandb.disable_artifact bool
                        Set to true to disable saving an artifact despite
                        training.save_checkpoint=True (default: False)
  --wandb.project str
  --wandb.entity [str]
  --wandb.notes [str]
  --wandb.run_id [str]
  --wandb.mode [str]    Allowed values: 'online', 'offline' 'disabled'.
                        Defaults to 'online' (default: None)
