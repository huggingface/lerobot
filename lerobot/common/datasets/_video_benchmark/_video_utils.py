"""This file contains work-in-progress alternative to default decoding strategy."""

import einops
import torch


def decode_video_frames_ffmpegio(video_path, timestamps, device="cpu"):
    # assert device == "cpu", f"Only CPU decoding is supported with ffmpegio, but device is {device}"
    import einops
    import ffmpegio

    num_contiguous_frames = 1  # noqa: F841
    image_format = "rgb24"

    list_frames = []
    for timestamp in timestamps:
        kwargs = {
            "ss": str(timestamp),
            # vframes=num_contiguous_frames,
            "pix_fmt": image_format,
            # hwaccel=None if device == "cpu" else device, # ,
            "show_log": True,
        }

        if device == "cuda":
            kwargs["hwaccel_in"] = "cuda"
            kwargs["hwaccel_output_format_in"] = "cuda"

        fs, frames = ffmpegio.video.read(str(video_path), **kwargs)
        list_frames.append(torch.from_numpy(frames))
    frames = torch.cat(list_frames)

    frames = einops.rearrange(frames, "b h w c -> b c h w")
    frames = frames.type(torch.float32) / 255
    return frames


def yuv_to_rgb(frames):
    assert frames.dtype == torch.uint8
    assert frames.ndim == 4
    assert frames.shape[1] == 3

    frames = frames.cpu().to(torch.float)
    y = frames[..., 0, :, :]
    u = frames[..., 1, :, :]
    v = frames[..., 2, :, :]

    y /= 255
    u = u / 255 - 0.5
    v = v / 255 - 0.5

    r = y + 1.13983 * v
    g = y + -0.39465 * u - 0.58060 * v
    b = y + 2.03211 * u

    rgb = torch.stack([r, g, b], 1)
    rgb = (rgb * 255).clamp(0, 255).to(torch.uint8)
    return rgb


def yuv_to_rgb_cv2(frames, return_hwc=True):
    assert frames.dtype == torch.uint8
    assert frames.ndim == 4
    assert frames.shape[1] == 3
    frames = frames.cpu()
    import cv2

    frames = einops.rearrange(frames, "b c h w -> b h w c")
    frames = frames.numpy()
    frames = [cv2.cvtColor(frame, cv2.COLOR_YUV2RGB) for frame in frames]
    frames = [torch.from_numpy(frame) for frame in frames]
    frames = torch.stack(frames)
    if not return_hwc:
        frames = einops.rearrange(frames, "b h w c -> b c h w")
    return frames


def decode_video_frames_torchaudio(video_path, timestamps, device="cpu"):
    num_contiguous_frames = 1
    width = None
    height = None
    # image_format = "rgb"  # or "yuv"
    # image_format = None
    image_format = "yuv444p"
    # image_format = "yuv444p"
    # image_format = "rgb24"
    frame_rate = None

    scale_full_range_filter = False

    filter_desc = []

    video_stream_kwgs = {
        "frames_per_chunk": num_contiguous_frames,
        # "buffer_chunk_size": num_contiguous_frames,
    }

    # choice of decoder
    if device == "cuda":
        video_stream_kwgs["hw_accel"] = "cuda:0"
        video_stream_kwgs["decoder"] = "h264_cuvid"
        # video_stream_kwgs["decoder"] = "hevc_cuvid"
        # video_stream_kwgs["decoder"] = "av1_cuvid"
        # video_stream_kwgs["decoder"] = "ffv1_cuvid"
    else:
        video_stream_kwgs["decoder"] = "h264"
        # video_stream_kwgs["decoder"] = "hevc"
        # video_stream_kwgs["decoder"] = "av1"
        # video_stream_kwgs["decoder"] = "ffv1"

    # resize
    resize_width = width is not None
    resize_height = height is not None
    if resize_width or resize_height:
        if device == "cuda":
            assert resize_width and resize_height
            video_stream_kwgs["decoder_option"] = {"resize": f"{width}x{height}"}
        else:
            scales = []
            if resize_width:
                scales.append(f"width={width}")
            if resize_height:
                scales.append(f"height={height}")
            filter_desc.append(f"scale={':'.join(scales)}")

    # choice of format
    if image_format is not None:
        if device == "cuda":
            # TODO(rcadene): rebuild ffmpeg with --enable-cuda-nvcc, --enable-cuvid, and --enable-libnpp
            # filter_desc.append(f"scale=format={image_format}")
            # filter_desc.append(f"scale_cuda=format={image_format}")
            # filter_desc.append(f"scale_npp=format={image_format}")
            filter_desc.append(f"format=pix_fmts={image_format}")
        else:
            filter_desc.append(f"format=pix_fmts={image_format}")

    # choice of frame rate
    if frame_rate is not None:
        filter_desc.append(f"fps={frame_rate}")

    # to set output scale [0-255] instead of [16-235]
    if scale_full_range_filter:
        filter_desc.append("scale=in_range=limited:out_range=full")

    if len(filter_desc) > 0:
        video_stream_kwgs["filter_desc"] = ",".join(filter_desc)

    # create a stream and load a certain number of frame at a certain frame rate
    # TODO(rcadene): make sure it's the most optimal way to do it
    from torchaudio.io import StreamReader

    print(video_stream_kwgs)

    list_frames = []
    for timestamp in timestamps:
        s = StreamReader(str(video_path))
        s.seek(timestamp)
        s.add_video_stream(**video_stream_kwgs)
        s.fill_buffer()
        (frames,) = s.pop_chunks()

        if "yuv" in image_format:
            frames = yuv_to_rgb(frames)

        assert frames.dtype == torch.uint8
        frames = frames.type(torch.float32)

        # if device == "cuda":
        # The original data had limited range, which is 16-235, and torchaudio does not convert,
        # while FFmpeg converts it to full range 0-255. So you can apply a linear transformation.
        if not scale_full_range_filter:
            frames -= 16
            frames *= 255 / (235 - 16)

        frames /= 255

        frames = frames.clip(0, 1)
        list_frames.append(frames)

    frames = torch.cat(list_frames)
    return frames


# def _decode_frames_decord(video_path, timestamp):
#     num_contiguous_frames = 1  # noqa: F841 TODO(rcadene): remove
#     device = "cpu"

#     from decord import VideoReader, cpu, gpu

#     with open(str(video_path), "rb") as f:
#         ctx = gpu if device == "cuda" else cpu
#         vr = VideoReader(f, ctx=ctx(0))  # noqa: F841
#         raise NotImplementedError("Convert `timestamp` into frame_id")
#     #     frame_id = frame_ids[0].item()
#     #     frames = vr.get_batch([frame_id])
#     # frames = torch.from_numpy(frames.asnumpy())
#     # frames = einops.rearrange(frames, "b h w c -> b c h w")
#     # return frames


# def decode_frames_nvc(video_path, timestamps, device="cuda"):
#     assert device == "cuda"

#     import PyNvCodec as nvc
#     import PytorchNvCodec as pnvc

#     gpuID = 0

#     nvDec = nvc.PyNvDecoder('path_to_video_file', gpuID)
#     to_rgb = nvc.PySurfaceConverter(nvDec.Width(), nvDec.Height(), nvc.PixelFormat.NV12, nvc.PixelFormat.RGB, gpuID)
#     to_planar = nvc.PySurfaceConverter(nvDec.Width(), nvDec.Height(), nvc.PixelFormat.RGB, nvc.PixelFormat.RGB_PLANAR, gpuID)

#     while True:
#         # Obtain NV12 decoded surface from decoder;
#         rawSurface = nvDec.DecodeSingleSurface()
#         if (rawSurface.Empty()):
#             break

#         # Convert to RGB interleaved;
#         rgb_byte = to_rgb.Execute(rawSurface)

#         # Convert to RGB planar because that's what to_tensor + normalize are doing;
#         rgb_planar = to_planar.Execute(rgb_byte)

#         # Create torch tensor from it and reshape because
#         # pnvc.makefromDevicePtrUint8 creates just a chunk of CUDA memory
#         # and then copies data from plane pointer to allocated chunk;
#         surfPlane = rgb_planar.PlanePtr()
#         surface_tensor = pnvc.makefromDevicePtrUint8(surfPlane.GpuMem(), surfPlane.Width(), surfPlane.Height(), surfPlane.Pitch(), surfPlane.ElemSize())
#         surface_tensor.resize_(3, target_h, target_w)
