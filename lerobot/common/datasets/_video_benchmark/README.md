# Video benchmark


## Questions

What is the optimal trade-off between:
- maximizing loading time with random access,
- minimizing memory space on disk,
- maximizing success rate of policies?

How to encode videos?
- How much compression (`-crf`)? Low compression with `0`, normal compression with `20` or extreme with `56`?
- What pixel format to use (`-pix_fmt`)? `yuv444p` or `yuv420p`?
- How many key frames (`-g`)? A key frame every `10` frames?

How to decode videos?
- Which `decoder`? `torchvision`, `torchaudio`, `ffmpegio`, `decord`, or `nvc`?

## Metrics

**Percentage of data compression (higher is better)**
`pc_compression` is the ratio of the memory space on disk taken by the original images to encode, to the memory space taken by the encoded video. For instance, `pc_compression=400%` means that the video takes 4 times less memory space on disk compared to the original images.

**Percentage of loading time (lower is better)**
`pc_load_time` is the ratio of the time it takes to load original images at given timestamps, to the time it takes to decode the exact same frames from the video. Lower is better. For instance, `pc_load_time=120%` means that decoding from video is a bit slower than loading the original images.

**Average L2 error per pixel (lower is better)**
`avg_per_pixel_l2_error` is the average L2 error between each decoded frame and its corresponding original image over all requested timestamps, and also divided by the number of pixels in the image to be comparable when switching to different image sizes.

**Loss of a pretrained policy (higher is better)** (not available)
`loss_pretrained` is the result of evaluating with the selected encoding/decoding settings a policy pretrained on original images. It is easier to understand than `avg_l2_error`.

**Success rate after retraining (higher is better)** (not available)
`success_rate` is the result of training and evaluating a policy with the selected encoding/decoding settings. It is the most difficult metric to get but also the very best.


## Variables

**Image content**
We don't expect the same optimal settings for a dataset of images from a simulation, or from real-world in an appartment, or in a factory, or outdoor, etc. Hence, we run this bechmark on two datasets: `pusht` (simulation) and `umi` (real-world outdoor).

**Requested timestamps**
In this benchmark, we focus on the loading time of random access, so we are not interested about sequentially loading all frames of a video like in a movie. However, the number of consecutive timestamps requested and their spacing can greatly affect the `pc_load_time`. In fact, it is expected to get faster loading time by decoding a large number of consecutive frames from a video, than to load the same data from individual images. To reflect our robotics use case, we consider a setting where we load 2 consecutive frames with 4 frames of spacing.

**Data augmentations**
We might revisit this benchmark and find better settings if we train our policies with various data augmentations to make them more robusts (e.g. robust to color changes, compression, etc.).


## Results

### Loading 2 consecutive frames with 4 frames spacing (Diffusion Policy setting)

**`decoder`**
| repo_id | decoder | pc_load_time | avg_per_pixel_l2_error |
| --- | --- | --- | --- |
| lerobot/pusht | <span style="color: #32CD32;">torchvision</span> | 0.166 | 0.0000119 |
| lerobot/pusht | ffmpegio | 0.009 | 0.0001182 |
| lerobot/pusht | torchaudio | 0.138 | 0.0000359 |
| lerobot/umi_cup_in_the_wild | <span style="color: #32CD32;">torchvision</span> | 0.174 | 0.0000174 |
| lerobot/umi_cup_in_the_wild | ffmpegio | 0.010 | 0.0000735 |
| lerobot/umi_cup_in_the_wild | torchaudio | 0.154 | 0.0000340 |

**`pix_fmt`**
| repo_id | pix_fmt | pc_compression | pc_load_time | avg_per_pixel_l2_error |
| --- | --- | --- | --- | --- |
| lerobot/pusht | yuv420p | 3.602 | 0.202 | 0.0000661 |
| lerobot/pusht | <span style="color: #32CD32;">yuv444p</span> | 3.213 | 0.153 | 0.0000110 |
| lerobot/umi_cup_in_the_wild | yuv420p | 8.879 | 0.202 | 0.0000332 |
| lerobot/umi_cup_in_the_wild | <span style="color: #32CD32;">yuv444p</span> | 8.517 | 0.165 | 0.0000175 |

**`g`**
| repo_id | g | pc_compression | pc_load_time | avg_per_pixel_l2_error |
| --- | --- | --- | --- | --- |
| lerobot/pusht | 1 | 1.308 | 0.190 | 0.0000151 |
| lerobot/pusht | 5 | 2.739 | 0.184 | 0.0000123 |
| lerobot/pusht | 10 | 3.213 | 0.144 | 0.0000116 |
| lerobot/pusht | 15 | 3.460 | 0.137 | 0.0000112 |
| lerobot/pusht | 20 | 3.559 | 0.118 | 0.0000109 |
| lerobot/pusht | 30 | 3.697 | 0.104 | 0.0000117 |
| lerobot/pusht | 40 | 3.763 | 0.092 | 0.0000116 |
| lerobot/pusht | 60 | 3.925 | 0.068 | 0.0000117 |
| lerobot/pusht | 100 | 4.010 | 0.054 | 0.0000117 |
| lerobot/pusht | <span style="color: #32CD32;">None</span> | 4.058 | 0.043 | 0.0000117 |
| lerobot/umi_cup_in_the_wild | 1 | 4.790 | 0.236 | 0.0000221 |
| lerobot/umi_cup_in_the_wild | 5 | 7.707 | 0.201 | 0.0000185 |
| lerobot/umi_cup_in_the_wild | 10 | 8.517 | 0.172 | 0.0000177 |
| lerobot/umi_cup_in_the_wild | 15 | 8.830 | 0.152 | 0.0000170 |
| lerobot/umi_cup_in_the_wild | 20 | 8.961 | 0.133 | 0.0000167 |
| lerobot/umi_cup_in_the_wild | 30 | 8.850 | 0.113 | 0.0000167 |
| lerobot/umi_cup_in_the_wild | 40 | 8.996 | 0.109 | 0.0000174 |
| lerobot/umi_cup_in_the_wild | 60 | 9.113 | 0.081 | 0.0000163 |
| lerobot/umi_cup_in_the_wild | 100 | 9.278 | 0.051 | 0.0000173 |
| lerobot/umi_cup_in_the_wild | <span style="color: #32CD32;">None</span> | 9.396 | 0.030 | 0.0000165 |

**`crf`**
| repo_id | crf | pc_compression | pc_load_time | avg_per_pixel_l2_error |
| --- | --- | --- | --- | --- |
| lerobot/pusht | 0 | 4.529 | 0.041 | 0.0000035 |
| lerobot/pusht | 5 | 3.138 | 0.040 | 0.0000077 |
| lerobot/pusht | <span style="color: #32CD32;">10</span> | 4.058 | 0.038 | 0.0000121 |
| lerobot/pusht | <span style="color: #32CD32;">15</span> | 5.407 | 0.039 | 0.0000195 |
| lerobot/pusht | <span style="color: #32CD32;">20</span> | 7.335 | 0.039 | 0.0000319 |
| lerobot/pusht | <span style="color: #32CD32;">None</span> | 8.909 | 0.046 | 0.0000425 |
| lerobot/pusht | 25 | 10.213 | 0.039 | 0.0000519 |
| lerobot/pusht | 30 | 14.516 | 0.041 | 0.0000795 |
| lerobot/pusht | 40 | 23.546 | 0.041 | 0.0001557 |
| lerobot/pusht | 50 | 28.460 | 0.042 | 0.0002723 |
| lerobot/umi_cup_in_the_wild | 0 | 2.318 | 0.012 | 0.0000056 |
| lerobot/umi_cup_in_the_wild | 5 | 4.899 | 0.019 | 0.0000132 |
| lerobot/umi_cup_in_the_wild | <span style="color: #32CD32;">10</span> | 9.396 | 0.026 | 0.0000183 |
| lerobot/umi_cup_in_the_wild | <span style="color: #32CD32;">15</span> | 19.161 | 0.034 | 0.0000241 |
| lerobot/umi_cup_in_the_wild | <span style="color: #32CD32;">20</span> | 39.311 | 0.039 | 0.0000329 |
| lerobot/umi_cup_in_the_wild | <span style="color: #32CD32;">None</span> | 60.530 | 0.043 | 0.0000401 |
| lerobot/umi_cup_in_the_wild | 25 | 81.048 | 0.046 | 0.0000454 |
| lerobot/umi_cup_in_the_wild | 30 | 165.189 | 0.051 | 0.0000609 |
| lerobot/umi_cup_in_the_wild | 40 | 544.478 | 0.056 | 0.0001095 |
| lerobot/umi_cup_in_the_wild | 50 | 1109.556 | 0.072 | 0.0001815 |


### Loading 6 consecutive frames with no spacing (TDMPC setting)

**`decoder`**
| repo_id | decoder | pc_load_time | avg_per_pixel_l2_error |
| --- | --- | --- | --- |
| lerobot/pusht | <span style="color: #32CD32;">torchvision</span> | 0.386 | 0.0000117 |
| lerobot/pusht | ffmpegio | 0.008 | 0.0000117 |
| lerobot/pusht | torchaudio | 0.184 | 0.0000356 |
| lerobot/umi_cup_in_the_wild | <span style="color: #32CD32;">torchvision</span> | 0.448 | 0.0000178 |
| lerobot/umi_cup_in_the_wild | ffmpegio | 0.009 | 0.0000178 |
| lerobot/umi_cup_in_the_wild | torchaudio | 0.149 | 0.0000349 |

**`pix_fmt`**
| repo_id | pix_fmt | pc_compression | pc_load_time | avg_per_pixel_l2_error |
| --- | --- | --- | --- | --- |
| lerobot/pusht | yuv420p | 3.602 | 0.518 | 0.0000651 |
| lerobot/pusht | <span style="color: #32CD32;">yuv444p</span> | 3.213 | 0.401 | 0.0000117 |
| lerobot/umi_cup_in_the_wild | yuv420p | 8.879 | 0.578 | 0.0000334 |
| lerobot/umi_cup_in_the_wild | <span style="color: #32CD32;">yuv444p</span> | 8.517 | 0.479 | 0.0000178 |

**`g`**
| repo_id | g | pc_compression | pc_load_time | avg_per_pixel_l2_error |
| --- | --- | --- | --- | --- |
| lerobot/pusht | 1 | 1.308 | 0.528 | 0.0000152 |
| lerobot/pusht | 5 | 2.739 | 0.483 | 0.0000124 |
| lerobot/pusht | 10 | 3.213 | 0.396 | 0.0000117 |
| lerobot/pusht | 15 | 3.460 | 0.379 | 0.0000118 |
| lerobot/pusht | 20 | 3.559 | 0.319 | 0.0000114 |
| lerobot/pusht | 30 | 3.697 | 0.278 | 0.0000116 |
| lerobot/pusht | 40 | 3.763 | 0.243 | 0.0000115 |
| lerobot/pusht | 60 | 3.925 | 0.186 | 0.0000118 |
| lerobot/pusht | 100 | 4.010 | 0.156 | 0.0000119 |
| lerobot/pusht | <span style="color: #32CD32;">None</span> | 4.058 | 0.105 | 0.0000121 |
| lerobot/umi_cup_in_the_wild | 1 | 4.790 | 0.605 | 0.0000221 |
| lerobot/umi_cup_in_the_wild | 5 | 7.707 | 0.533 | 0.0000183 |
| lerobot/umi_cup_in_the_wild | 10 | 8.517 | 0.469 | 0.0000178 |
| lerobot/umi_cup_in_the_wild | 15 | 8.830 | 0.399 | 0.0000174 |
| lerobot/umi_cup_in_the_wild | 20 | 8.961 | 0.382 | 0.0000175 |
| lerobot/umi_cup_in_the_wild | 30 | 8.850 | 0.326 | 0.0000172 |
| lerobot/umi_cup_in_the_wild | 40 | 8.996 | 0.279 | 0.0000173 |
| lerobot/umi_cup_in_the_wild | 60 | 9.113 | 0.226 | 0.0000174 |
| lerobot/umi_cup_in_the_wild | 100 | 9.278 | 0.150 | 0.0000175 |
| lerobot/umi_cup_in_the_wild | <span style="color: #32CD32;">None</span> | 9.396 | 0.076 | 0.0000176 |

**`crf`**
| repo_id | crf | pc_compression | pc_load_time | avg_per_pixel_l2_error |
| --- | --- | --- | --- | --- |
| lerobot/pusht | 0 | 4.529 | 0.108 | 0.0000035 |
| lerobot/pusht | 5 | 3.138 | 0.099 | 0.0000077 |
| lerobot/pusht | 10 | 4.058 | 0.091 | 0.0000121 |
| lerobot/pusht | 15 | 5.407 | 0.095 | 0.0000195 |
| lerobot/pusht | 20 | 7.335 | 0.100 | 0.0000318 |
| lerobot/pusht | <span style="color: #32CD32;">None</span> | 8.909 | 0.102 | 0.0000422 |
| lerobot/pusht | 25 | 10.213 | 0.102 | 0.0000517 |
| lerobot/pusht | 30 | 14.516 | 0.104 | 0.0000795 |
| lerobot/pusht | 40 | 23.546 | 0.106 | 0.0001555 |
| lerobot/pusht | 50 | 28.460 | 0.110 | 0.0002723 |
| lerobot/umi_cup_in_the_wild | 0 | 2.318 | 0.032 | 0.0000056 |
| lerobot/umi_cup_in_the_wild | 5 | 4.899 | 0.052 | 0.0000127 |
| lerobot/umi_cup_in_the_wild | <span style="color: #32CD32;">10</span> | 9.396 | 0.073 | 0.0000176 |
| lerobot/umi_cup_in_the_wild | <span style="color: #32CD32;">15</span> | 19.161 | 0.097 | 0.0000234 |
| lerobot/umi_cup_in_the_wild | <span style="color: #32CD32;">20</span> | 39.311 | 0.110 | 0.0000321 |
| lerobot/umi_cup_in_the_wild | <span style="color: #32CD32;">None</span> | 60.530 | 0.117 | 0.0000393 |
| lerobot/umi_cup_in_the_wild | 25 | 81.048 | 0.126 | 0.0000446 |
| lerobot/umi_cup_in_the_wild | 30 | 165.189 | 0.138 | 0.0000603 |
| lerobot/umi_cup_in_the_wild | 40 | 544.478 | 0.151 | 0.0001095 |
| lerobot/umi_cup_in_the_wild | 50 | 1109.556 | 0.167 | 0.0001817 |
