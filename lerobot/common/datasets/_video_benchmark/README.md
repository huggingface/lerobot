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
`compression_factor` is the ratio of the memory space on disk taken by the original images to encode, to the memory space taken by the encoded video. For instance, `compression_factor=4` means that the video takes 4 times less memory space on disk compared to the original images.

**Percentage of loading time (higher is better)**
`load_time_factor` is the ratio of the time it takes to load original images at given timestamps, to the time it takes to decode the exact same frames from the video. Higher is better. For instance, `load_time_factor=0.5` means that decoding from video is 2 times slower than loading the original images.

**Average L2 error per pixel (lower is better)**
`avg_per_pixel_l2_error` is the average L2 error between each decoded frame and its corresponding original image over all requested timestamps, and also divided by the number of pixels in the image to be comparable when switching to different image sizes.

**Loss of a pretrained policy (higher is better)** (not available)
`loss_pretrained` is the result of evaluating with the selected encoding/decoding settings a policy pretrained on original images. It is easier to understand than `avg_l2_error`.

**Success rate after retraining (higher is better)** (not available)
`success_rate` is the result of training and evaluating a policy with the selected encoding/decoding settings. It is the most difficult metric to get but also the very best.


## Variables

**Image content**
We don't expect the same optimal settings for a dataset of images from a simulation, or from real-world in an appartment, or in a factory, or outdoor, etc. Hence, we run this benchmark on two datasets: `pusht` (simulation) and `umi` (real-world outdoor).

**Requested timestamps**
In this benchmark, we focus on the loading time of random access, so we are not interested in sequentially loading all frames of a video like in a movie. However, the number of consecutive timestamps requested and their spacing can greatly affect the `load_time_factor`. In fact, it is expected to get faster loading time by decoding a large number of consecutive frames from a video, than to load the same data from individual images. To reflect our robotics use case, we consider a few settings:
- `single_frame`: 1 frame,
- `2_frames`: 2 consecutive frames (e.g. `[t, t + 1 / fps]`),
- `2_frames_4_space`: 2 consecutive frames with 4 frames of spacing (e.g `[t, t + 4 / fps]`),

**Data augmentations**
We might revisit this benchmark and find better settings if we train our policies with various data augmentations to make them more robust (e.g. robust to color changes, compression, etc.).


## Results

**`decoder`**
| repo_id | decoder | load_time_factor | avg_per_pixel_l2_error |
| --- | --- | --- | --- |
| lerobot/pusht | <span style="color: #32CD32;">torchvision</span> | 0.166 | 0.0000119 |
| lerobot/pusht | ffmpegio | 0.009 | 0.0001182 |
| lerobot/pusht | torchaudio | 0.138 | 0.0000359 |
| lerobot/umi_cup_in_the_wild | <span style="color: #32CD32;">torchvision</span> | 0.174 | 0.0000174 |
| lerobot/umi_cup_in_the_wild | ffmpegio | 0.010 | 0.0000735 |
| lerobot/umi_cup_in_the_wild | torchaudio | 0.154 | 0.0000340 |

### `1_frame`

**`pix_fmt`**
| repo_id | pix_fmt | compression_factor | load_time_factor | avg_per_pixel_l2_error |
| --- | --- | --- | --- | --- |
| lerobot/pusht | yuv420p | 3.788 | 0.224 | 0.0000760 |
| lerobot/pusht | yuv444p | 3.646 | 0.185 | 0.0000443 |
| lerobot/umi_cup_in_the_wild | yuv420p | 14.391 | 0.388 | 0.0000469 |
| lerobot/umi_cup_in_the_wild | yuv444p | 14.932 | 0.329 | 0.0000397 |

**`g`**
| repo_id | g | compression_factor | load_time_factor | avg_per_pixel_l2_error |
| --- | --- | --- | --- | --- |
| lerobot/pusht | 1 | 2.543 | 0.204 | 0.0000556 |
| lerobot/pusht | 2 | 3.646 | 0.182 | 0.0000443 |
| lerobot/pusht | 3 | 4.431 | 0.174 | 0.0000450 |
| lerobot/pusht | 4 | 5.103 | 0.163 | 0.0000448 |
| lerobot/pusht | 5 | 5.625 | 0.163 | 0.0000436 |
| lerobot/pusht | 6 | 5.974 | 0.155 | 0.0000427 |
| lerobot/pusht | 10 | 6.814 | 0.130 | 0.0000410 |
| lerobot/pusht | 15 | 7.431 | 0.105 | 0.0000406 |
| lerobot/pusht | 20 | 7.662 | 0.097 | 0.0000400 |
| lerobot/pusht | 40 | 8.163 | 0.061 | 0.0000405 |
| lerobot/pusht | 100 | 8.761 | 0.039 | 0.0000422 |
| lerobot/pusht | None | 8.909 | 0.024 | 0.0000431 |
| lerobot/umi_cup_in_the_wild | 1 | 14.411 | 0.444 | 0.0000601 |
| lerobot/umi_cup_in_the_wild | 2 | 14.932 | 0.345 | 0.0000397 |
| lerobot/umi_cup_in_the_wild | 3 | 20.174 | 0.282 | 0.0000416 |
| lerobot/umi_cup_in_the_wild | 4 | 24.889 | 0.271 | 0.0000415 |
| lerobot/umi_cup_in_the_wild | 5 | 28.825 | 0.260 | 0.0000415 |
| lerobot/umi_cup_in_the_wild | 6 | 31.635 | 0.249 | 0.0000415 |
| lerobot/umi_cup_in_the_wild | 10 | 39.418 | 0.195 | 0.0000399 |
| lerobot/umi_cup_in_the_wild | 15 | 44.577 | 0.169 | 0.0000394 |
| lerobot/umi_cup_in_the_wild | 20 | 47.907 | 0.140 | 0.0000390 |
| lerobot/umi_cup_in_the_wild | 40 | 52.554 | 0.096 | 0.0000384 |
| lerobot/umi_cup_in_the_wild | 100 | 58.241 | 0.046 | 0.0000390 |
| lerobot/umi_cup_in_the_wild | None | 60.530 | 0.022 | 0.0000400 |

**`crf`**
| repo_id | crf | compression_factor | load_time_factor | avg_per_pixel_l2_error |
| --- | --- | --- | --- | --- |
| lerobot/pusht | 0 | 1.699 | 0.175 | 0.0000035 |
| lerobot/pusht | 5 | 1.409 | 0.181 | 0.0000080 |
| lerobot/pusht | 10 | 1.842 | 0.172 | 0.0000123 |
| lerobot/pusht | 15 | 2.322 | 0.187 | 0.0000211 |
| lerobot/pusht | 20 | 3.050 | 0.181 | 0.0000346 |
| lerobot/pusht | None | 3.646 | 0.189 | 0.0000443 |
| lerobot/pusht | 25 | 3.969 | 0.186 | 0.0000521 |
| lerobot/pusht | 30 | 5.687 | 0.184 | 0.0000850 |
| lerobot/pusht | 40 | 10.818 | 0.193 | 0.0001726 |
| lerobot/pusht | 50 | 18.185 | 0.183 | 0.0002606 |
| lerobot/umi_cup_in_the_wild | 0 | 1.918 | 0.165 | 0.0000056 |
| lerobot/umi_cup_in_the_wild | 5 | 3.207 | 0.171 | 0.0000111 |
| lerobot/umi_cup_in_the_wild | 10 | 4.818 | 0.212 | 0.0000153 |
| lerobot/umi_cup_in_the_wild | 15 | 7.329 | 0.261 | 0.0000218 |
| lerobot/umi_cup_in_the_wild | 20 | 11.361 | 0.312 | 0.0000317 |
| lerobot/umi_cup_in_the_wild | None | 14.932 | 0.339 | 0.0000397 |
| lerobot/umi_cup_in_the_wild | 25 | 17.741 | 0.297 | 0.0000452 |
| lerobot/umi_cup_in_the_wild | 30 | 27.983 | 0.406 | 0.0000629 |
| lerobot/umi_cup_in_the_wild | 40 | 82.449 | 0.468 | 0.0001184 |
| lerobot/umi_cup_in_the_wild | 50 | 186.145 | 0.515 | 0.0001879 |

**best**
| repo_id | compression_factor | load_time_factor | avg_per_pixel_l2_error |
| --- | --- | --- | --- |
| lerobot/pusht | 3.646 | 0.188 | 0.0000443 |
| lerobot/umi_cup_in_the_wild | 14.932 | 0.339 | 0.0000397 |

### `2_frames`

**`pix_fmt`**
| repo_id | pix_fmt | compression_factor | load_time_factor | avg_per_pixel_l2_error |
| --- | --- | --- | --- | --- |
| lerobot/pusht | yuv420p | 3.788 | 0.314 | 0.0000799 |
| lerobot/pusht | yuv444p | 3.646 | 0.303 | 0.0000496 |
| lerobot/umi_cup_in_the_wild | yuv420p | 14.391 | 0.642 | 0.0000503 |
| lerobot/umi_cup_in_the_wild | yuv444p | 14.932 | 0.529 | 0.0000436 |

**`g`**
| repo_id | g | compression_factor | load_time_factor | avg_per_pixel_l2_error |
| --- | --- | --- | --- | --- |
| lerobot/pusht | 1 | 2.543 | 0.308 | 0.0000599 |
| lerobot/pusht | 2 | 3.646 | 0.279 | 0.0000496 |
| lerobot/pusht | 3 | 4.431 | 0.259 | 0.0000498 |
| lerobot/pusht | 4 | 5.103 | 0.243 | 0.0000501 |
| lerobot/pusht | 5 | 5.625 | 0.235 | 0.0000492 |
| lerobot/pusht | 6 | 5.974 | 0.230 | 0.0000481 |
| lerobot/pusht | 10 | 6.814 | 0.194 | 0.0000468 |
| lerobot/pusht | 15 | 7.431 | 0.152 | 0.0000460 |
| lerobot/pusht | 20 | 7.662 | 0.151 | 0.0000455 |
| lerobot/pusht | 40 | 8.163 | 0.095 | 0.0000454 |
| lerobot/pusht | 100 | 8.761 | 0.062 | 0.0000472 |
| lerobot/pusht | None | 8.909 | 0.037 | 0.0000479 |
| lerobot/umi_cup_in_the_wild | 1 | 14.411 | 0.638 | 0.0000625 |
| lerobot/umi_cup_in_the_wild | 2 | 14.932 | 0.537 | 0.0000436 |
| lerobot/umi_cup_in_the_wild | 3 | 20.174 | 0.493 | 0.0000437 |
| lerobot/umi_cup_in_the_wild | 4 | 24.889 | 0.458 | 0.0000446 |
| lerobot/umi_cup_in_the_wild | 5 | 28.825 | 0.438 | 0.0000445 |
| lerobot/umi_cup_in_the_wild | 6 | 31.635 | 0.424 | 0.0000444 |
| lerobot/umi_cup_in_the_wild | 10 | 39.418 | 0.345 | 0.0000435 |
| lerobot/umi_cup_in_the_wild | 15 | 44.577 | 0.313 | 0.0000417 |
| lerobot/umi_cup_in_the_wild | 20 | 47.907 | 0.264 | 0.0000421 |
| lerobot/umi_cup_in_the_wild | 40 | 52.554 | 0.185 | 0.0000414 |
| lerobot/umi_cup_in_the_wild | 100 | 58.241 | 0.090 | 0.0000420 |
| lerobot/umi_cup_in_the_wild | None | 60.530 | 0.042 | 0.0000424 |

**`crf`**
| repo_id | crf | compression_factor | load_time_factor | avg_per_pixel_l2_error |
| --- | --- | --- | --- | --- |
| lerobot/pusht | 0 | 1.699 | 0.302 | 0.0000097 |
| lerobot/pusht | 5 | 1.409 | 0.287 | 0.0000142 |
| lerobot/pusht | 10 | 1.842 | 0.283 | 0.0000184 |
| lerobot/pusht | 15 | 2.322 | 0.305 | 0.0000268 |
| lerobot/pusht | 20 | 3.050 | 0.285 | 0.0000402 |
| lerobot/pusht | None | 3.646 | 0.285 | 0.0000496 |
| lerobot/pusht | 25 | 3.969 | 0.293 | 0.0000572 |
| lerobot/pusht | 30 | 5.687 | 0.293 | 0.0000893 |
| lerobot/pusht | 40 | 10.818 | 0.319 | 0.0001762 |
| lerobot/pusht | 50 | 18.185 | 0.304 | 0.0002626 |
| lerobot/umi_cup_in_the_wild | 0 | 1.918 | 0.235 | 0.0000112 |
| lerobot/umi_cup_in_the_wild | 5 | 3.207 | 0.261 | 0.0000166 |
| lerobot/umi_cup_in_the_wild | 10 | 4.818 | 0.333 | 0.0000207 |
| lerobot/umi_cup_in_the_wild | 15 | 7.329 | 0.406 | 0.0000267 |
| lerobot/umi_cup_in_the_wild | 20 | 11.361 | 0.489 | 0.0000361 |
| lerobot/umi_cup_in_the_wild | None | 14.932 | 0.537 | 0.0000436 |
| lerobot/umi_cup_in_the_wild | 25 | 17.741 | 0.578 | 0.0000487 |
| lerobot/umi_cup_in_the_wild | 30 | 27.983 | 0.453 | 0.0000655 |
| lerobot/umi_cup_in_the_wild | 40 | 82.449 | 0.767 | 0.0001192 |
| lerobot/umi_cup_in_the_wild | 50 | 186.145 | 0.816 | 0.0001881 |

**best**
| repo_id | compression_factor | load_time_factor | avg_per_pixel_l2_error |
| --- | --- | --- | --- |
| lerobot/pusht | 3.646 | 0.283 | 0.0000496 |
| lerobot/umi_cup_in_the_wild | 14.932 | 0.543 | 0.0000436 |

### `2_frames_4_space`

**`pix_fmt`**
| repo_id | pix_fmt | compression_factor | load_time_factor | avg_per_pixel_l2_error |
| --- | --- | --- | --- | --- |
| lerobot/pusht | yuv420p | 3.788 | 0.257 | 0.0000855 |
| lerobot/pusht | yuv444p | 3.646 | 0.261 | 0.0000556 |
| lerobot/umi_cup_in_the_wild | yuv420p | 14.391 | 0.493 | 0.0000476 |
| lerobot/umi_cup_in_the_wild | yuv444p | 14.932 | 0.371 | 0.0000404 |

**`g`**
| repo_id | g | compression_factor | load_time_factor | avg_per_pixel_l2_error |
| --- | --- | --- | --- | --- |
| lerobot/pusht | 1 | 2.543 | 0.226 | 0.0000670 |
| lerobot/pusht | 2 | 3.646 | 0.222 | 0.0000556 |
| lerobot/pusht | 3 | 4.431 | 0.217 | 0.0000567 |
| lerobot/pusht | 4 | 5.103 | 0.204 | 0.0000555 |
| lerobot/pusht | 5 | 5.625 | 0.179 | 0.0000556 |
| lerobot/pusht | 6 | 5.974 | 0.188 | 0.0000544 |
| lerobot/pusht | 10 | 6.814 | 0.160 | 0.0000531 |
| lerobot/pusht | 15 | 7.431 | 0.150 | 0.0000521 |
| lerobot/pusht | 20 | 7.662 | 0.123 | 0.0000519 |
| lerobot/pusht | 40 | 8.163 | 0.092 | 0.0000519 |
| lerobot/pusht | 100 | 8.761 | 0.053 | 0.0000533 |
| lerobot/pusht | None | 8.909 | 0.034 | 0.0000541 |
| lerobot/umi_cup_in_the_wild | 1 | 14.411 | 0.409 | 0.0000607 |
| lerobot/umi_cup_in_the_wild | 2 | 14.932 | 0.381 | 0.0000404 |
| lerobot/umi_cup_in_the_wild | 3 | 20.174 | 0.355 | 0.0000418 |
| lerobot/umi_cup_in_the_wild | 4 | 24.889 | 0.346 | 0.0000425 |
| lerobot/umi_cup_in_the_wild | 5 | 28.825 | 0.354 | 0.0000419 |
| lerobot/umi_cup_in_the_wild | 6 | 31.635 | 0.336 | 0.0000419 |
| lerobot/umi_cup_in_the_wild | 10 | 39.418 | 0.314 | 0.0000402 |
| lerobot/umi_cup_in_the_wild | 15 | 44.577 | 0.269 | 0.0000397 |
| lerobot/umi_cup_in_the_wild | 20 | 47.907 | 0.246 | 0.0000395 |
| lerobot/umi_cup_in_the_wild | 40 | 52.554 | 0.171 | 0.0000390 |
| lerobot/umi_cup_in_the_wild | 100 | 58.241 | 0.091 | 0.0000399 |
| lerobot/umi_cup_in_the_wild | None | 60.530 | 0.043 | 0.0000409 |

**`crf`**
| repo_id | crf | compression_factor | load_time_factor | avg_per_pixel_l2_error |
| --- | --- | --- | --- | --- |
| lerobot/pusht | 0 | 1.699 | 0.212 | 0.0000193 |
| lerobot/pusht | 5 | 1.409 | 0.211 | 0.0000232 |
| lerobot/pusht | 10 | 1.842 | 0.199 | 0.0000270 |
| lerobot/pusht | 15 | 2.322 | 0.198 | 0.0000347 |
| lerobot/pusht | 20 | 3.050 | 0.211 | 0.0000469 |
| lerobot/pusht | None | 3.646 | 0.206 | 0.0000556 |
| lerobot/pusht | 25 | 3.969 | 0.210 | 0.0000626 |
| lerobot/pusht | 30 | 5.687 | 0.223 | 0.0000927 |
| lerobot/pusht | 40 | 10.818 | 0.227 | 0.0001763 |
| lerobot/pusht | 50 | 18.185 | 0.223 | 0.0002625 |
| lerobot/umi_cup_in_the_wild | 0 | 1.918 | 0.147 | 0.0000071 |
| lerobot/umi_cup_in_the_wild | 5 | 3.207 | 0.182 | 0.0000125 |
| lerobot/umi_cup_in_the_wild | 10 | 4.818 | 0.222 | 0.0000166 |
| lerobot/umi_cup_in_the_wild | 15 | 7.329 | 0.270 | 0.0000229 |
| lerobot/umi_cup_in_the_wild | 20 | 11.361 | 0.325 | 0.0000326 |
| lerobot/umi_cup_in_the_wild | None | 14.932 | 0.362 | 0.0000404 |
| lerobot/umi_cup_in_the_wild | 25 | 17.741 | 0.390 | 0.0000459 |
| lerobot/umi_cup_in_the_wild | 30 | 27.983 | 0.437 | 0.0000633 |
| lerobot/umi_cup_in_the_wild | 40 | 82.449 | 0.499 | 0.0001186 |
| lerobot/umi_cup_in_the_wild | 50 | 186.145 | 0.564 | 0.0001879 |

**best**
| repo_id | compression_factor | load_time_factor | avg_per_pixel_l2_error |
| --- | --- | --- | --- |
| lerobot/pusht | 3.646 | 0.224 | 0.0000556 |
| lerobot/umi_cup_in_the_wild | 14.932 | 0.368 | 0.0000404 |

### `6_frames`

**`pix_fmt`**
| repo_id | pix_fmt | compression_factor | load_time_factor | avg_per_pixel_l2_error |
| --- | --- | --- | --- | --- |
| lerobot/pusht | yuv420p | 3.788 | 0.660 | 0.0000839 |
| lerobot/pusht | yuv444p | 3.646 | 0.546 | 0.0000542 |
| lerobot/umi_cup_in_the_wild | yuv420p | 14.391 | 1.225 | 0.0000497 |
| lerobot/umi_cup_in_the_wild | yuv444p | 14.932 | 0.908 | 0.0000428 |

**`g`**
| repo_id | g | compression_factor | load_time_factor | avg_per_pixel_l2_error |
| --- | --- | --- | --- | --- |
| lerobot/pusht | 1 | 2.543 | 0.552 | 0.0000646 |
| lerobot/pusht | 2 | 3.646 | 0.534 | 0.0000542 |
| lerobot/pusht | 3 | 4.431 | 0.563 | 0.0000546 |
| lerobot/pusht | 4 | 5.103 | 0.537 | 0.0000545 |
| lerobot/pusht | 5 | 5.625 | 0.477 | 0.0000532 |
| lerobot/pusht | 6 | 5.974 | 0.515 | 0.0000530 |
| lerobot/pusht | 10 | 6.814 | 0.410 | 0.0000512 |
| lerobot/pusht | 15 | 7.431 | 0.405 | 0.0000503 |
| lerobot/pusht | 20 | 7.662 | 0.345 | 0.0000500 |
| lerobot/pusht | 40 | 8.163 | 0.247 | 0.0000496 |
| lerobot/pusht | 100 | 8.761 | 0.147 | 0.0000510 |
| lerobot/pusht | None | 8.909 | 0.100 | 0.0000519 |
| lerobot/umi_cup_in_the_wild | 1 | 14.411 | 0.997 | 0.0000620 |
| lerobot/umi_cup_in_the_wild | 2 | 14.932 | 0.911 | 0.0000428 |
| lerobot/umi_cup_in_the_wild | 3 | 20.174 | 0.869 | 0.0000433 |
| lerobot/umi_cup_in_the_wild | 4 | 24.889 | 0.874 | 0.0000438 |
| lerobot/umi_cup_in_the_wild | 5 | 28.825 | 0.864 | 0.0000439 |
| lerobot/umi_cup_in_the_wild | 6 | 31.635 | 0.834 | 0.0000440 |
| lerobot/umi_cup_in_the_wild | 10 | 39.418 | 0.781 | 0.0000421 |
| lerobot/umi_cup_in_the_wild | 15 | 44.577 | 0.679 | 0.0000411 |
| lerobot/umi_cup_in_the_wild | 20 | 47.907 | 0.652 | 0.0000410 |
| lerobot/umi_cup_in_the_wild | 40 | 52.554 | 0.465 | 0.0000404 |
| lerobot/umi_cup_in_the_wild | 100 | 58.241 | 0.245 | 0.0000413 |
| lerobot/umi_cup_in_the_wild | None | 60.530 | 0.116 | 0.0000417 |

**`crf`**
| repo_id | crf | compression_factor | load_time_factor | avg_per_pixel_l2_error |
| --- | --- | --- | --- | --- |
| lerobot/pusht | 0 | 1.699 | 0.534 | 0.0000163 |
| lerobot/pusht | 5 | 1.409 | 0.524 | 0.0000205 |
| lerobot/pusht | 10 | 1.842 | 0.510 | 0.0000245 |
| lerobot/pusht | 15 | 2.322 | 0.512 | 0.0000324 |
| lerobot/pusht | 20 | 3.050 | 0.508 | 0.0000452 |
| lerobot/pusht | None | 3.646 | 0.518 | 0.0000542 |
| lerobot/pusht | 25 | 3.969 | 0.534 | 0.0000616 |
| lerobot/pusht | 30 | 5.687 | 0.530 | 0.0000927 |
| lerobot/pusht | 40 | 10.818 | 0.552 | 0.0001777 |
| lerobot/pusht | 50 | 18.185 | 0.564 | 0.0002644 |
| lerobot/umi_cup_in_the_wild | 0 | 1.918 | 0.401 | 0.0000101 |
| lerobot/umi_cup_in_the_wild | 5 | 3.207 | 0.499 | 0.0000156 |
| lerobot/umi_cup_in_the_wild | 10 | 4.818 | 0.599 | 0.0000197 |
| lerobot/umi_cup_in_the_wild | 15 | 7.329 | 0.704 | 0.0000258 |
| lerobot/umi_cup_in_the_wild | 20 | 11.361 | 0.834 | 0.0000352 |
| lerobot/umi_cup_in_the_wild | None | 14.932 | 0.925 | 0.0000428 |
| lerobot/umi_cup_in_the_wild | 25 | 17.741 | 0.978 | 0.0000480 |
| lerobot/umi_cup_in_the_wild | 30 | 27.983 | 1.088 | 0.0000648 |
| lerobot/umi_cup_in_the_wild | 40 | 82.449 | 1.324 | 0.0001190 |
| lerobot/umi_cup_in_the_wild | 50 | 186.145 | 1.436 | 0.0001880 |

**best**
| repo_id | compression_factor | load_time_factor | avg_per_pixel_l2_error |
| --- | --- | --- | --- |
| lerobot/pusht | 3.646 | 0.546 | 0.0000542 |
| lerobot/umi_cup_in_the_wild | 14.932 | 0.934 | 0.0000428 |
