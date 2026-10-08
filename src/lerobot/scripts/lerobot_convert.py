# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""`lerobot-convert`: one source recipe, shared resumable conversion runtime."""

from lerobot.configs import parser
from lerobot.data_processing.conversion import ConvertConfig, convert_dataset


@parser.wrap()
def convert(cfg: ConvertConfig):
    if cfg.runtime.backend == "hf_jobs":
        from lerobot.jobs.processing import submit_convert_to_hf

        return submit_convert_to_hf(cfg)
    convert_dataset(cfg)


def main():
    convert()
