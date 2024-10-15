#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# imagecodecs/numcodecs.py

# Copyright (c) 2021-2022, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Copied from: https://github.com/real-stanford/universal_manipulation_interface/blob/298776ce251f33b6b3185a98d6e7d1f9ad49168b/diffusion_policy/codecs/imagecodecs_numcodecs.py#L1
"""Additional numcodecs implemented using imagecodecs."""

__version__ = "2022.9.26"

__all__ = ("register_codecs",)

import imagecodecs
import numpy
from numcodecs.abc import Codec
from numcodecs.registry import get_codec, register_codec

# TODO (azouitine): Remove useless codecs


def protective_squeeze(x: numpy.ndarray):
    """
    Squeeze dim only if it's not the last dim.
    Image dim expected to be *, H, W, C
    """
    img_shape = x.shape[-3:]
    if len(x.shape) > 3:
        n_imgs = numpy.prod(x.shape[:-3])
        if n_imgs > 1:
            img_shape = (-1,) + img_shape
    return x.reshape(img_shape)


def get_default_image_compressor(**kwargs):
    if imagecodecs.JPEGXL:
        # has JPEGXL
        this_kwargs = {
            "effort": 3,
            "distance": 0.3,
            # bug in libjxl, invalid codestream for non-lossless
            # when decoding speed > 1
            "decodingspeed": 1,
        }
        this_kwargs.update(kwargs)
        return JpegXl(**this_kwargs)
    else:
        this_kwargs = {"level": 50}
        this_kwargs.update(kwargs)
        return Jpeg2k(**this_kwargs)


class Jpeg2k(Codec):
    """JPEG 2000 codec for numcodecs."""

    codec_id = "imagecodecs_jpeg2k"

    def __init__(
        self,
        level=None,
        codecformat=None,
        colorspace=None,
        tile=None,
        reversible=None,
        bitspersample=None,
        resolutions=None,
        numthreads=None,
        verbose=0,
    ):
        self.level = level
        self.codecformat = codecformat
        self.colorspace = colorspace
        self.tile = None if tile is None else tuple(tile)
        self.reversible = reversible
        self.bitspersample = bitspersample
        self.resolutions = resolutions
        self.numthreads = numthreads
        self.verbose = verbose

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))
        return imagecodecs.jpeg2k_encode(
            buf,
            level=self.level,
            codecformat=self.codecformat,
            colorspace=self.colorspace,
            tile=self.tile,
            reversible=self.reversible,
            bitspersample=self.bitspersample,
            resolutions=self.resolutions,
            numthreads=self.numthreads,
            verbose=self.verbose,
        )

    def decode(self, buf, out=None):
        return imagecodecs.jpeg2k_decode(buf, verbose=self.verbose, numthreads=self.numthreads, out=out)


class JpegXl(Codec):
    """JPEG XL codec for numcodecs."""

    codec_id = "imagecodecs_jpegxl"

    def __init__(
        self,
        # encode
        level=None,
        effort=None,
        distance=None,
        lossless=None,
        decodingspeed=None,
        photometric=None,
        planar=None,
        usecontainer=None,
        # decode
        index=None,
        keeporientation=None,
        # both
        numthreads=None,
    ):
        """
        Return JPEG XL image from numpy array.
        Float must be in nominal range 0..1.

        Currently L, LA, RGB, RGBA images are supported in contig mode.
        Extra channels are only supported for grayscale images in planar mode.

        Parameters
        ----------
        level : Default to None, i.e. not overwriting lossess and decodingspeed options.
            When < 0: Use lossless compression
            When in [0,1,2,3,4]: Sets the decoding speed tier for the provided options.
                Minimum is 0 (slowest to decode, best quality/density), and maximum
                is 4 (fastest to decode, at the cost of some quality/density).
        effort : Default to 3.
            Sets encoder effort/speed level without affecting decoding speed.
            Valid values are, from faster to slower speed: 1:lightning 2:thunder
                3:falcon 4:cheetah 5:hare 6:wombat 7:squirrel 8:kitten 9:tortoise.
            Speed: lightning, thunder, falcon, cheetah, hare, wombat, squirrel, kitten, tortoise
            control the encoder effort in ascending order.
            This also affects memory usage: using lower effort will typically reduce memory
            consumption during encoding.
            lightning and thunder are fast modes useful for lossless mode (modular).
            falcon disables all of the following tools.
            cheetah enables coefficient reordering, context clustering, and heuristics for selecting DCT sizes and quantization steps.
            hare enables Gaborish filtering, chroma from luma, and an initial estimate of quantization steps.
            wombat enables error diffusion quantization and full DCT size selection heuristics.
            squirrel (default) enables dots, patches, and spline detection, and full context clustering.
            kitten optimizes the adaptive quantization for a psychovisual metric.
            tortoise enables a more thorough adaptive quantization search.
        distance : Default to 1.0
            Sets the distance level for lossy compression: target max butteraugli distance,
            lower = higher quality. Range: 0 .. 15. 0.0 = mathematically lossless
            (however, use JxlEncoderSetFrameLossless instead to use true lossless,
            as setting distance to 0 alone is not the only requirement).
            1.0 = visually lossless. Recommended range: 0.5 .. 3.0.
        lossess : Default to False.
            Use lossess encoding.
        decodingspeed : Default to 0.
            Duplicate to level. [0,4]
        photometric : Return JxlColorSpace value.
            Default logic is quite complicated but works most of the time.
            Accepted value:
                int: [-1,3]
                str: ['RGB',
                    'WHITEISZERO', 'MINISWHITE',
                    'BLACKISZERO', 'MINISBLACK', 'GRAY',
                    'XYB', 'KNOWN']
        planar : Enable multi-channel mode.
            Default to false.
        usecontainer :
            Forces the encoder to use the box-based container format (BMFF)
            even when not necessary.
            When using JxlEncoderUseBoxes, JxlEncoderStoreJPEGMetadata or
            JxlEncoderSetCodestreamLevel with level 10, the encoder will
            automatically also use the container format, it is not necessary
            to use JxlEncoderUseContainer for those use cases.
            By default this setting is disabled.
        index : Selectively decode frames for animation.
            Default to 0, decode all frames.
            When set to > 0, decode that frame index only.
        keeporientation :
            Enables or disables preserving of as-in-bitstream pixeldata orientation.
            Some images are encoded with an Orientation tag indicating that the
            decoder must perform a rotation and/or mirroring to the encoded image data.

            If skip_reorientation is JXL_FALSE (the default): the decoder will apply
            the transformation from the orientation setting, hence rendering the image
            according to its specified intent. When producing a JxlBasicInfo, the decoder
            will always set the orientation field to JXL_ORIENT_IDENTITY (matching the
            returned pixel data) and also align xsize and ysize so that they correspond
            to the width and the height of the returned pixel data.

            If skip_reorientation is JXL_TRUE: the decoder will skip applying the
            transformation from the orientation setting, returning the image in
            the as-in-bitstream pixeldata orientation. This may be faster to decode
            since the decoder doesnt have to apply the transformation, but can
            cause wrong display of the image if the orientation tag is not correctly
            taken into account by the user.

            By default, this option is disabled, and the returned pixel data is
            re-oriented according to the images Orientation setting.
        threads : Default to 1.
            If <= 0, use all cores.
            If > 32, clipped to 32.
        """

        self.level = level
        self.effort = effort
        self.distance = distance
        self.lossless = bool(lossless)
        self.decodingspeed = decodingspeed
        self.photometric = photometric
        self.planar = planar
        self.usecontainer = usecontainer
        self.index = index
        self.keeporientation = keeporientation
        self.numthreads = numthreads

    def encode(self, buf):
        # TODO: only squeeze all but last dim
        buf = protective_squeeze(numpy.asarray(buf))
        return imagecodecs.jpegxl_encode(
            buf,
            level=self.level,
            effort=self.effort,
            distance=self.distance,
            lossless=self.lossless,
            decodingspeed=self.decodingspeed,
            photometric=self.photometric,
            planar=self.planar,
            usecontainer=self.usecontainer,
            numthreads=self.numthreads,
        )

    def decode(self, buf, out=None):
        return imagecodecs.jpegxl_decode(
            buf,
            index=self.index,
            keeporientation=self.keeporientation,
            numthreads=self.numthreads,
            out=out,
        )


def _flat(out):
    """Return numpy array as contiguous view of bytes if possible."""
    if out is None:
        return None
    view = memoryview(out)
    if view.readonly or not view.contiguous:
        return None
    return view.cast("B")


def register_codecs(codecs=None, force=False, verbose=True):
    """Register codecs in this module with numcodecs."""
    for name, cls in globals().items():
        if not hasattr(cls, "codec_id") or name == "Codec":
            continue
        if codecs is not None and cls.codec_id not in codecs:
            continue
        try:
            try:  # noqa: SIM105
                get_codec({"id": cls.codec_id})
            except TypeError:
                # registered, but failed
                pass
        except ValueError:
            # not registered yet
            pass
        else:
            if not force:
                if verbose:
                    log_warning(f"numcodec {cls.codec_id!r} already registered")
                continue
            if verbose:
                log_warning(f"replacing registered numcodec {cls.codec_id!r}")
        register_codec(cls)


def log_warning(msg, *args, **kwargs):
    """Log message with level WARNING."""
    import logging

    logging.getLogger(__name__).warning(msg, *args, **kwargs)
