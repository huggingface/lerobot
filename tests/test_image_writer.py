import queue
import time
from multiprocessing import queues
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from lerobot.common.datasets.image_writer import (
    AsyncImageWriter,
    image_array_to_pil_image,
    safe_stop_image_writer,
    write_image,
)
from tests.fixtures.constants import DUMMY_HWC

DUMMY_IMAGE = "test_image.png"


def test_init_threading():
    writer = AsyncImageWriter(num_processes=0, num_threads=2)
    try:
        assert writer.num_processes == 0
        assert writer.num_threads == 2
        assert isinstance(writer.queue, queue.Queue)
        assert len(writer.threads) == 2
        assert len(writer.processes) == 0
        assert all(t.is_alive() for t in writer.threads)
    finally:
        writer.stop()


def test_init_multiprocessing():
    writer = AsyncImageWriter(num_processes=2, num_threads=2)
    try:
        assert writer.num_processes == 2
        assert writer.num_threads == 2
        assert isinstance(writer.queue, queues.JoinableQueue)
        assert len(writer.threads) == 0
        assert len(writer.processes) == 2
        assert all(p.is_alive() for p in writer.processes)
    finally:
        writer.stop()


def test_zero_threads():
    with pytest.raises(ValueError):
        AsyncImageWriter(num_processes=0, num_threads=0)


def test_image_array_to_pil_image_float_array_wrong_range_0_255():
    image = np.random.rand(*DUMMY_HWC) * 255
    with pytest.raises(ValueError):
        image_array_to_pil_image(image)


def test_image_array_to_pil_image_float_array_wrong_range_neg_1_1():
    image = np.random.rand(*DUMMY_HWC) * 2 - 1
    with pytest.raises(ValueError):
        image_array_to_pil_image(image)


def test_image_array_to_pil_image_rgb(img_array_factory):
    img_array = img_array_factory(100, 100)
    result_image = image_array_to_pil_image(img_array)
    assert isinstance(result_image, Image.Image)
    assert result_image.size == (100, 100)
    assert result_image.mode == "RGB"


def test_image_array_to_pil_image_pytorch_format(img_array_factory):
    img_array = img_array_factory(100, 100).transpose(2, 0, 1)
    result_image = image_array_to_pil_image(img_array)
    assert isinstance(result_image, Image.Image)
    assert result_image.size == (100, 100)
    assert result_image.mode == "RGB"


def test_image_array_to_pil_image_single_channel(img_array_factory):
    img_array = img_array_factory(channels=1)
    with pytest.raises(NotImplementedError):
        image_array_to_pil_image(img_array)


def test_image_array_to_pil_image_4_channels(img_array_factory):
    img_array = img_array_factory(channels=4)
    with pytest.raises(NotImplementedError):
        image_array_to_pil_image(img_array)


def test_image_array_to_pil_image_float_array(img_array_factory):
    img_array = img_array_factory(dtype=np.float32)
    result_image = image_array_to_pil_image(img_array)
    assert isinstance(result_image, Image.Image)
    assert result_image.size == (100, 100)
    assert result_image.mode == "RGB"
    assert np.array(result_image).dtype == np.uint8


def test_image_array_to_pil_image_uint8_array(img_array_factory):
    img_array = img_array_factory(dtype=np.float32)
    result_image = image_array_to_pil_image(img_array)
    assert isinstance(result_image, Image.Image)
    assert result_image.size == (100, 100)
    assert result_image.mode == "RGB"
    assert np.array(result_image).dtype == np.uint8


def test_write_image_numpy(tmp_path, img_array_factory):
    image_array = img_array_factory()
    fpath = tmp_path / DUMMY_IMAGE
    write_image(image_array, fpath)
    assert fpath.exists()
    saved_image = np.array(Image.open(fpath))
    assert np.array_equal(image_array, saved_image)


def test_write_image_image(tmp_path, img_factory):
    image_pil = img_factory()
    fpath = tmp_path / DUMMY_IMAGE
    write_image(image_pil, fpath)
    assert fpath.exists()
    saved_image = Image.open(fpath)
    assert list(saved_image.getdata()) == list(image_pil.getdata())
    assert np.array_equal(image_pil, saved_image)


def test_write_image_exception(tmp_path):
    image_array = "invalid data"
    fpath = tmp_path / DUMMY_IMAGE
    with patch("builtins.print") as mock_print:
        write_image(image_array, fpath)
        mock_print.assert_called()
        assert not fpath.exists()


def test_save_image_numpy(tmp_path, img_array_factory):
    writer = AsyncImageWriter()
    try:
        image_array = img_array_factory()
        fpath = tmp_path / DUMMY_IMAGE
        fpath.parent.mkdir(parents=True, exist_ok=True)
        writer.save_image(image_array, fpath)
        writer.wait_until_done()
        assert fpath.exists()
        saved_image = np.array(Image.open(fpath))
        assert np.array_equal(image_array, saved_image)
    finally:
        writer.stop()


def test_save_image_numpy_multiprocessing(tmp_path, img_array_factory):
    writer = AsyncImageWriter(num_processes=2, num_threads=2)
    try:
        image_array = img_array_factory()
        fpath = tmp_path / DUMMY_IMAGE
        writer.save_image(image_array, fpath)
        writer.wait_until_done()
        assert fpath.exists()
        saved_image = np.array(Image.open(fpath))
        assert np.array_equal(image_array, saved_image)
    finally:
        writer.stop()


def test_save_image_torch(tmp_path, img_tensor_factory):
    writer = AsyncImageWriter()
    try:
        image_tensor = img_tensor_factory()
        fpath = tmp_path / DUMMY_IMAGE
        fpath.parent.mkdir(parents=True, exist_ok=True)
        writer.save_image(image_tensor, fpath)
        writer.wait_until_done()
        assert fpath.exists()
        saved_image = np.array(Image.open(fpath))
        expected_image = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        assert np.array_equal(expected_image, saved_image)
    finally:
        writer.stop()


def test_save_image_torch_multiprocessing(tmp_path, img_tensor_factory):
    writer = AsyncImageWriter(num_processes=2, num_threads=2)
    try:
        image_tensor = img_tensor_factory()
        fpath = tmp_path / DUMMY_IMAGE
        writer.save_image(image_tensor, fpath)
        writer.wait_until_done()
        assert fpath.exists()
        saved_image = np.array(Image.open(fpath))
        expected_image = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        assert np.array_equal(expected_image, saved_image)
    finally:
        writer.stop()


def test_save_image_pil(tmp_path, img_factory):
    writer = AsyncImageWriter()
    try:
        image_pil = img_factory()
        fpath = tmp_path / DUMMY_IMAGE
        fpath.parent.mkdir(parents=True, exist_ok=True)
        writer.save_image(image_pil, fpath)
        writer.wait_until_done()
        assert fpath.exists()
        saved_image = Image.open(fpath)
        assert list(saved_image.getdata()) == list(image_pil.getdata())
    finally:
        writer.stop()


def test_save_image_pil_multiprocessing(tmp_path, img_factory):
    writer = AsyncImageWriter(num_processes=2, num_threads=2)
    try:
        image_pil = img_factory()
        fpath = tmp_path / DUMMY_IMAGE
        writer.save_image(image_pil, fpath)
        writer.wait_until_done()
        assert fpath.exists()
        saved_image = Image.open(fpath)
        assert list(saved_image.getdata()) == list(image_pil.getdata())
    finally:
        writer.stop()


def test_save_image_invalid_data(tmp_path):
    writer = AsyncImageWriter()
    try:
        image_array = "invalid data"
        fpath = tmp_path / DUMMY_IMAGE
        fpath.parent.mkdir(parents=True, exist_ok=True)
        with patch("builtins.print") as mock_print:
            writer.save_image(image_array, fpath)
            writer.wait_until_done()
            mock_print.assert_called()
            assert not fpath.exists()
    finally:
        writer.stop()


def test_save_image_after_stop(tmp_path, img_array_factory):
    writer = AsyncImageWriter()
    writer.stop()
    image_array = img_array_factory()
    fpath = tmp_path / DUMMY_IMAGE
    writer.save_image(image_array, fpath)
    time.sleep(1)
    assert not fpath.exists()


def test_stop():
    writer = AsyncImageWriter(num_processes=0, num_threads=2)
    writer.stop()
    assert not any(t.is_alive() for t in writer.threads)


def test_stop_multiprocessing():
    writer = AsyncImageWriter(num_processes=2, num_threads=2)
    writer.stop()
    assert not any(p.is_alive() for p in writer.processes)


def test_multiple_stops():
    writer = AsyncImageWriter()
    writer.stop()
    writer.stop()  # Should not raise an exception
    assert not any(t.is_alive() for t in writer.threads)


def test_multiple_stops_multiprocessing():
    writer = AsyncImageWriter(num_processes=2, num_threads=2)
    writer.stop()
    writer.stop()  # Should not raise an exception
    assert not any(t.is_alive() for t in writer.threads)


def test_wait_until_done(tmp_path, img_array_factory):
    writer = AsyncImageWriter(num_processes=0, num_threads=4)
    try:
        num_images = 100
        image_arrays = [img_array_factory(height=500, width=500) for _ in range(num_images)]
        fpaths = [tmp_path / f"frame_{i:06d}.png" for i in range(num_images)]
        for image_array, fpath in zip(image_arrays, fpaths, strict=True):
            fpath.parent.mkdir(parents=True, exist_ok=True)
            writer.save_image(image_array, fpath)
        writer.wait_until_done()
        for i, fpath in enumerate(fpaths):
            assert fpath.exists()
            saved_image = np.array(Image.open(fpath))
            assert np.array_equal(saved_image, image_arrays[i])
    finally:
        writer.stop()


def test_wait_until_done_multiprocessing(tmp_path, img_array_factory):
    writer = AsyncImageWriter(num_processes=2, num_threads=2)
    try:
        num_images = 100
        image_arrays = [img_array_factory() for _ in range(num_images)]
        fpaths = [tmp_path / f"frame_{i:06d}.png" for i in range(num_images)]
        for image_array, fpath in zip(image_arrays, fpaths, strict=True):
            fpath.parent.mkdir(parents=True, exist_ok=True)
            writer.save_image(image_array, fpath)
        writer.wait_until_done()
        for i, fpath in enumerate(fpaths):
            assert fpath.exists()
            saved_image = np.array(Image.open(fpath))
            assert np.array_equal(saved_image, image_arrays[i])
    finally:
        writer.stop()


def test_exception_handling(tmp_path, img_array_factory):
    writer = AsyncImageWriter()
    try:
        image_array = img_array_factory()
        with (
            patch.object(writer.queue, "put", side_effect=queue.Full("Queue is full")),
            pytest.raises(queue.Full) as exc_info,
        ):
            writer.save_image(image_array, tmp_path / "test.png")
        assert str(exc_info.value) == "Queue is full"
    finally:
        writer.stop()


def test_with_different_image_formats(tmp_path, img_array_factory):
    writer = AsyncImageWriter()
    try:
        image_array = img_array_factory()
        formats = ["png", "jpeg", "bmp"]
        for fmt in formats:
            fpath = tmp_path / f"test_image.{fmt}"
            write_image(image_array, fpath)
            assert fpath.exists()
    finally:
        writer.stop()


def test_safe_stop_image_writer_decorator():
    class MockDataset:
        def __init__(self):
            self.image_writer = MagicMock(spec=AsyncImageWriter)

    @safe_stop_image_writer
    def function_that_raises_exception(dataset=None):
        raise Exception("Test exception")

    dataset = MockDataset()

    with pytest.raises(Exception) as exc_info:
        function_that_raises_exception(dataset=dataset)

    assert str(exc_info.value) == "Test exception"
    dataset.image_writer.stop.assert_called_once()


def test_main_process_time(tmp_path, img_tensor_factory):
    writer = AsyncImageWriter()
    try:
        image_tensor = img_tensor_factory()
        fpath = tmp_path / DUMMY_IMAGE
        start_time = time.perf_counter()
        writer.save_image(image_tensor, fpath)
        end_time = time.perf_counter()
        time_spent = end_time - start_time
        # Might need to adjust this threshold depending on hardware
        assert time_spent < 0.01, f"Main process time exceeded threshold: {time_spent}s"
        writer.wait_until_done()
        assert fpath.exists()
    finally:
        writer.stop()
