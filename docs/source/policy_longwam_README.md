<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Long-WAM

Long-WAM predicts robot actions using an extended observation history.

- [Project](https://nvlabs.github.io/LongLive/Long-WAM/)
- [Reference implementation](https://github.com/NVlabs/LongLive/tree/main/Long-WAM)
- [Integration guide](longwam.mdx)

Use `policy.type=longwam` for native training and per-tick `select_action` for
inference. IDM and COD support Long-WAM's synchronous/asynchronous scheduling
and optional streaming VAE encoding. Existing release bundles require the
provided converter. See the integration guide for commands and validation.
