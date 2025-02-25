# Video benchmark


## Questions
What is the optimal trade-off between:
- maximizing loading time with random access,
- minimizing memory space on disk,
- maximizing success rate of policies,
- compatibility across devices/platforms for decoding videos (e.g. video players, web browsers).

How to encode videos?
- Which video codec (`-vcodec`) to use? h264, h265, AV1?
- What pixel format to use (`-pix_fmt`)? `yuv444p` or `yuv420p`?
- How much compression (`-crf`)? No compression with `0`, intermediate compression with `25` or extreme with `50+`?
- Which frequency to chose for key frames (`-g`)? A key frame every `10` frames?

How to decode videos?
- Which `decoder`? `torchvision`, `torchaudio`, `ffmpegio`, `decord`, or `nvc`?
- What scenarios to use for the requesting timestamps during benchmark? (`timestamps_mode`)


## Variables
**Image content & size**
We don't expect the same optimal settings for a dataset of images from a simulation, or from real-world in an apartment, or in a factory, or outdoor, or with lots of moving objects in the scene, etc. Similarly, loading times might not vary linearly with the image size (resolution).
For these reasons, we run this benchmark on four representative datasets:
- `lerobot/pusht_image`: (96 x 96 pixels) simulation with simple geometric shapes, fixed camera.
- `aliberts/aloha_mobile_shrimp_image`: (480 x 640 pixels) real-world indoor, moving camera.
- `aliberts/paris_street`: (720 x 1280 pixels) real-world outdoor, moving camera.
- `aliberts/kitchen`: (1080 x 1920 pixels) real-world indoor, fixed camera.

Note: The datasets used for this benchmark need to be image datasets, not video datasets.

**Data augmentations**
We might revisit this benchmark and find better settings if we train our policies with various data augmentations to make them more robust (e.g. robust to color changes, compression, etc.).

### Encoding parameters
| parameter   | values                                                       |
|-------------|--------------------------------------------------------------|
| **vcodec**  | `libx264`, `libx265`, `libsvtav1`                            |
| **pix_fmt** | `yuv444p`, `yuv420p`                                         |
| **g**       | `1`, `2`, `3`, `4`, `5`, `6`, `10`, `15`, `20`, `40`, `None` |
| **crf**     | `0`, `5`, `10`, `15`, `20`, `25`, `30`, `40`, `50`, `None`   |

Note that `crf` value might be interpreted differently by various video codecs. In other words, the same value used with one codec doesn't necessarily translate into the same compression level with another codec. In fact, the default value (`None`) isn't the same amongst the different video codecs. Importantly, it is also the case for many other ffmpeg arguments like `g` which specifies the frequency of the key frames.

For a comprehensive list and documentation of these parameters, see the ffmpeg documentation depending on the video codec used:
- h264: https://trac.ffmpeg.org/wiki/Encode/H.264
- h265: https://trac.ffmpeg.org/wiki/Encode/H.265
- AV1: https://trac.ffmpeg.org/wiki/Encode/AV1

### Decoding parameters
**Decoder**
We tested two video decoding backends from torchvision:
- `pyav` (default)
- `video_reader` (requires to build torchvision from source)

**Requested timestamps**
Given the way video decoding works, once a keyframe has been loaded, the decoding of subsequent frames is fast.
This of course is affected by the `-g` parameter during encoding, which specifies the frequency of the keyframes. Given our typical use cases in robotics policies which might request a few timestamps in different random places, we want to replicate these use cases with the following scenarios:
- `1_frame`: 1 frame,
- `2_frames`: 2 consecutive frames (e.g. `[t, t + 1 / fps]`),
- `6_frames`: 6 consecutive frames (e.g. `[t + i / fps for i in range(6)]`)

Note that this differs significantly from a typical use case like watching a movie, in which every frame is loaded sequentially from the beginning to the end and it's acceptable to have big values for `-g`.

Additionally, because some policies might request single timestamps that are a few frames apart, we also have the following scenario:
- `2_frames_4_space`: 2 frames with 4 consecutive frames of spacing in between (e.g `[t, t + 5 / fps]`),

However, due to how video decoding is implemented with `pyav`, we don't have access to an accurate seek so in practice this scenario is essentially the same as `6_frames` since all 6 frames between `t` and `t + 5 / fps` will be decoded.


## Metrics
**Data compression ratio (lower is better)**
`video_images_size_ratio` is the ratio of the memory space on disk taken by the encoded video over the memory space taken by the original images. For instance, `video_images_size_ratio=25%` means that the video takes 4 times less memory space on disk compared to the original images.

**Loading time ratio (lower is better)**
`video_images_load_time_ratio` is the ratio of the time it takes to decode frames from the video at a given timestamps over the time it takes to load the exact same original images. Lower is better. For instance, `video_images_load_time_ratio=200%` means that decoding from video is 2 times slower than loading the original images.

**Average Mean Square Error (lower is better)**
`avg_mse` is the average mean square error between each decoded frame and its corresponding original image over all requested timestamps, and also divided by the number of pixels in the image to be comparable when switching to different image sizes.

**Average Peak Signal to Noise Ratio (higher is better)**
`avg_psnr` measures the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation. Higher PSNR indicates better quality.

**Average Structural Similarity Index Measure (higher is better)**
`avg_ssim` evaluates the perceived quality of images by comparing luminance, contrast, and structure. SSIM values range from -1 to 1, where 1 indicates perfect similarity.

One aspect that can't be measured here with those metrics is the compatibility of the encoding across platforms, in particular on web browser, for visualization purposes.
h264, h265 and AV1 are all commonly used codecs and should not pose an issue. However, the chroma subsampling (`pix_fmt`) format might affect compatibility:
- `yuv420p` is more widely supported across various platforms, including web browsers.
- `yuv444p` offers higher color fidelity but might not be supported as broadly.


<!-- **Loss of a pretrained policy (higher is better)** (not available)
`loss_pretrained` is the result of evaluating with the selected encoding/decoding settings a policy pretrained on original images. It is easier to understand than `avg_l2_error`.

**Success rate after retraining (higher is better)** (not available)
`success_rate` is the result of training and evaluating a policy with the selected encoding/decoding settings. It is the most difficult metric to get but also the very best. -->


## How the benchmark works
The benchmark evaluates both encoding and decoding of video frames on the first episode of each dataset.

**Encoding:** for each `vcodec` and `pix_fmt` pair, we use a default value for `g` and `crf` upon which we change a single value (either `g` or `crf`) to one of the specified values (we don't test every combination of those as this would be computationally too heavy).
This gives a unique set of encoding parameters which is used to encode the episode.

**Decoding:** Then, for each of those unique encodings, we iterate through every combination of the decoding parameters `backend` and `timestamps_mode`. For each of them, we record the metrics of a number of samples (given by `--num-samples`). This is parallelized for efficiency and the number of processes can be controlled with `--num-workers`. Ideally, it's best to have a `--num-samples` that is divisible by `--num-workers`.

Intermediate results saved for each `vcodec` and `pix_fmt` combination in csv tables.
These are then all concatenated to a single table ready for analysis.

## Caveats
We tried to measure the most impactful parameters for both encoding and decoding. However, for computational reasons we can't test out every combination.

Additional encoding parameters exist that are not included in this benchmark. In particular:
- `-preset` which allows for selecting encoding presets. This represents a collection of options that will provide a certain encoding speed to compression ratio. By leaving this parameter unspecified, it is considered to be `medium` for libx264 and libx265 and `8` for libsvtav1.
- `-tune` which allows to optimize the encoding for certain aspects (e.g. film quality, fast decoding, etc.).

See the documentation mentioned above for more detailed info on these settings and for a more comprehensive list of other parameters.

Similarly on the decoding side, other decoders exist but are not implemented in our current benchmark. To name a few:
- `torchaudio`
- `ffmpegio`
- `decord`
- `nvc`

Note as well that since we are mostly interested in the performance at decoding time (also because encoding is done only once before uploading a dataset), we did not measure encoding times nor have any metrics regarding encoding.
However, besides the necessity to build ffmpeg from source, encoding did not pose any issue and it didn't take a significant amount of time during this benchmark.


## Install
Building ffmpeg from source is required to include libx265 and libaom/libsvtav1 (av1) video codecs ([compilation guide](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu)).

**Note:** While you still need to build torchvision with a conda-installed `ffmpeg<4.3` to use the `video_reader` decoder (as described in [#220](https://github.com/huggingface/lerobot/pull/220)), you also need another version which is custom-built with all the video codecs for encoding. For the script to then use that version, you can prepend the command above with `PATH="$HOME/bin:$PATH"`, which is where ffmpeg should be built.


## Adding a video decoder
Right now, we're only benchmarking the two video decoder available with torchvision: `pyav` and `video_reader`.
You can easily add a new decoder to benchmark by adding it to this function in the script:
```diff
def decode_video_frames(
    video_path: str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str,
) -> torch.Tensor:
    if backend in ["pyav", "video_reader"]:
        return decode_video_frames_torchvision(
            video_path, timestamps, tolerance_s, backend
        )
+    elif backend == ["your_decoder"]:
+        return your_decoder_function(
+            video_path, timestamps, tolerance_s, backend
+        )
    else:
        raise NotImplementedError(backend)
```


## Example
For a quick run, you can try these parameters:
```bash
python benchmark/video/run_video_benchmark.py \
    --output-dir outputs/video_benchmark \
    --repo-ids \
        lerobot/pusht_image \
        aliberts/aloha_mobile_shrimp_image \
    --vcodec libx264 libx265 \
    --pix-fmt yuv444p yuv420p \
    --g 2 20 None \
    --crf 10 40 None \
    --timestamps-modes 1_frame 2_frames \
    --backends pyav video_reader \
    --num-samples 5 \
    --num-workers 5 \
    --save-frames 0
```


## Results

### Reproduce
We ran the benchmark with the following parameters:
```bash
# h264 and h265 encodings
python benchmark/video/run_video_benchmark.py \
    --output-dir outputs/video_benchmark \
    --repo-ids \
        lerobot/pusht_image \
        aliberts/aloha_mobile_shrimp_image \
        aliberts/paris_street \
        aliberts/kitchen \
    --vcodec libx264 libx265 \
    --pix-fmt yuv444p yuv420p \
    --g 1 2 3 4 5 6 10 15 20 40 None \
    --crf 0 5 10 15 20 25 30 40 50 None \
    --timestamps-modes 1_frame 2_frames 6_frames \
    --backends pyav video_reader \
    --num-samples 50 \
    --num-workers 5 \
    --save-frames 1

# av1 encoding (only compatible with yuv420p and pyav decoder)
python benchmark/video/run_video_benchmark.py \
    --output-dir outputs/video_benchmark \
    --repo-ids \
        lerobot/pusht_image \
        aliberts/aloha_mobile_shrimp_image \
        aliberts/paris_street \
        aliberts/kitchen \
    --vcodec libsvtav1 \
    --pix-fmt yuv420p \
    --g 1 2 3 4 5 6 10 15 20 40 None \
    --crf 0 5 10 15 20 25 30 40 50 None \
    --timestamps-modes 1_frame 2_frames 6_frames \
    --backends pyav \
    --num-samples 50 \
    --num-workers 5 \
    --save-frames 1
```

The full results are available [here](https://docs.google.com/spreadsheets/d/1OYJB43Qu8fC26k_OyoMFgGBBKfQRCi4BIuYitQnq3sw/edit?usp=sharing)


### Parameters selected for LeRobotDataset
Considering these results, we chose what we think is the best set of encoding parameter:
- vcodec: `libsvtav1`
- pix-fmt: `yuv420p`
- g: `2`
- crf: `30`

Since we're using av1 encoding, we're choosing the `pyav` decoder as `video_reader` does not support it (and `pyav` doesn't require a custom build of `torchvision`).

### Summary

These tables show the results for `g=2` and `crf=30`, using `timestamps-modes=6_frames` and `backend=pyav`

| video_images_size_ratio            | vcodec     | pix_fmt |           |           |           |
|------------------------------------|------------|---------|-----------|-----------|-----------|
|                                    | libx264    |         | libx265   |           | libsvtav1 |
| repo_id                            | yuv420p    | yuv444p | yuv420p   | yuv444p   | yuv420p   |
| lerobot/pusht_image                | **16.97%** | 17.58%  | 18.57%    | 18.86%    | 22.06%    |
| aliberts/aloha_mobile_shrimp_image | 2.14%      | 2.11%   | 1.38%     | **1.37%** | 5.59%     |
| aliberts/paris_street              | 2.12%      | 2.13%   | **1.54%** | **1.54%** | 4.43%     |
| aliberts/kitchen                   | 1.40%      | 1.39%   | **1.00%** | **1.00%** | 2.52%     |

| video_images_load_time_ratio       | vcodec  | pix_fmt |          |         |           |
|------------------------------------|---------|---------|----------|---------|-----------|
|                                    | libx264 |         | libx265  |         | libsvtav1 |
| repo_id                            | yuv420p | yuv444p | yuv420p  | yuv444p | yuv420p   |
| lerobot/pusht_image                | 6.45    | 5.19    | **1.90** | 2.12    | 2.47      |
| aliberts/aloha_mobile_shrimp_image | 11.80   | 7.92    | 0.71     | 0.85    | **0.48**  |
| aliberts/paris_street              | 2.21    | 2.05    | 0.36     | 0.49    | **0.30**  |
| aliberts/kitchen                   | 1.46    | 1.46    | 0.28     | 0.51    | **0.26**  |

|                                    |          | vcodec   | pix_fmt      |          |           |              |
|------------------------------------|----------|----------|--------------|----------|-----------|--------------|
|                                    |          | libx264  |              | libx265  |           | libsvtav1    |
| repo_id                            | metric   | yuv420p  | yuv444p      | yuv420p  | yuv444p   | yuv420p      |
| lerobot/pusht_image                | avg_mse  | 2.90E-04 | **2.03E-04** | 3.13E-04 | 2.29E-04  | 2.19E-04     |
|                                    | avg_psnr | 35.44    | 37.07        | 35.49    | **37.30** | 37.20        |
|                                    | avg_ssim | 98.28%   | **98.85%**   | 98.31%   | 98.84%    | 98.72%       |
| aliberts/aloha_mobile_shrimp_image | avg_mse  | 2.76E-04 | 2.59E-04     | 3.17E-04 | 3.06E-04  | **1.30E-04** |
|                                    | avg_psnr | 35.91    | 36.21        | 35.88    | 36.09     | **40.17**    |
|                                    | avg_ssim | 95.19%   | 95.18%       | 95.00%   | 95.05%    | **97.73%**   |
| aliberts/paris_street              | avg_mse  | 6.89E-04 | 6.70E-04     | 4.03E-03 | 4.02E-03  | **3.09E-04** |
|                                    | avg_psnr | 33.48    | 33.68        | 32.05    | 32.15     | **35.40**    |
|                                    | avg_ssim | 93.76%   | 93.75%       | 89.46%   | 89.46%    | **95.46%**   |
| aliberts/kitchen                   | avg_mse  | 2.50E-04 | 2.24E-04     | 4.28E-04 | 4.18E-04  | **1.53E-04** |
|                                    | avg_psnr | 36.73    | 37.33        | 36.56    | 36.75     | **39.12**    |
|                                    | avg_ssim | 95.47%   | 95.58%       | 95.52%   | 95.53%    | **96.82%**   |
