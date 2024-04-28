import warnings

import imageio


def write_video(video_path, stacked_frames, fps):
    # Filter out DeprecationWarnings raised from pkg_resources
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "pkg_resources is deprecated as an API", category=DeprecationWarning
        )
        imageio.mimsave(video_path, stacked_frames, fps=fps)
