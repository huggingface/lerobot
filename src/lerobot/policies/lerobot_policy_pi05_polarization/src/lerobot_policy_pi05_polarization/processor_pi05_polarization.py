from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

def make_pi05_polarization_pre_post_processors(config, dataset_stats=None):
    # reuse the stock pi05 pipeline for images/state/action/language unchanged —
    # the polarization keys pass through it untouched (not declared as needing
    # quantile/mean-std normalization), and get consumed directly inside
    # embed_prefix's build_polfem_inputs call instead.
    return make_pi05_pre_post_processors(config, dataset_stats=dataset_stats)