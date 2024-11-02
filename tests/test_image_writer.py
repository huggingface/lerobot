import queue
import time
from multiprocessing import queues
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from lerobot.common.datasets.image_writer import (
    ImageWriter,
    image_array_to_image,
    safe_stop_image_writer,
    write_image,
)

DUMMY_IMAGE = "test_image.png"


def test_init_threading(tmp_path):
    writer = ImageWriter(write_dir=tmp_path, num_processes=0, num_threads=2)
    try:
        assert writer.num_processes == 0
        assert writer.num_threads == 2
        assert isinstance(writer.queue, queue.Queue)
        assert len(writer.threads) == 2
        assert len(writer.processes) == 0
        assert all(t.is_alive() for t in writer.threads)
    finally:
        writer.stop()


def test_init_multiprocessing(tmp_path):
    writer = ImageWriter(write_dir=tmp_path, num_processes=2, num_threads=2)
    try:
        assert writer.num_processes == 2
        assert writer.num_threads == 2
        assert isinstance(writer.queue, queues.JoinableQueue)
        assert len(writer.threads) == 0
        assert len(writer.processes) == 2
        assert all(p.is_alive() for p in writer.processes)
    finally:
        writer.stop()


def test_write_dir_created(tmp_path):
    write_dir = tmp_path / "non_existent_dir"
    assert not write_dir.exists()
    writer = ImageWriter(write_dir=write_dir)
    try:
        assert write_dir.exists()
    finally:
        writer.stop()


def test_get_image_file_path_and_episode_dir(tmp_path):
    writer = ImageWriter(write_dir=tmp_path)
    try:
        episode_index = 1
        image_key = "test_key"
        frame_index = 10
        expected_episode_dir = tmp_path / f"{image_key}/episode_{episode_index:06d}"
        expected_path = expected_episode_dir / f"frame_{frame_index:06d}.png"
        image_file_path = writer.get_image_file_path(episode_index, image_key, frame_index)
        assert image_file_path == expected_path
        episode_dir = writer.get_episode_dir(episode_index, image_key)
        assert episode_dir == expected_episode_dir
    finally:
        writer.stop()


def test_zero_threads(tmp_path):
    with pytest.raises(ValueError):
        ImageWriter(write_dir=tmp_path, num_processes=0, num_threads=0)


def test_image_array_to_image_rgb(img_array_factory):
    img_array = img_array_factory(100, 100)
    result_image = image_array_to_image(img_array)
    assert isinstance(result_image, Image.Image)
    assert result_image.size == (100, 100)
    assert result_image.mode == "RGB"


def test_image_array_to_image_pytorch_format(img_array_factory):
    img_array = img_array_factory(100, 100).transpose(2, 0, 1)
    result_image = image_array_to_image(img_array)
    assert isinstance(result_image, Image.Image)
    assert result_image.size == (100, 100)
    assert result_image.mode == "RGB"


@pytest.mark.skip("TODO: implement")
def test_image_array_to_image_single_channel(img_array_factory):
    img_array = img_array_factory(channels=1)
    result_image = image_array_to_image(img_array)
    assert isinstance(result_image, Image.Image)
    assert result_image.size == (100, 100)
    assert result_image.mode == "L"


def test_image_array_to_image_float_array(img_array_factory):
    img_array = img_array_factory(dtype=np.float32)
    result_image = image_array_to_image(img_array)
    assert isinstance(result_image, Image.Image)
    assert result_image.size == (100, 100)
    assert result_image.mode == "RGB"
    assert np.array(result_image).dtype == np.uint8


def test_image_array_to_image_out_of_bounds_float():
    # Float array with values out of [0, 1]
    img_array = np.random.uniform(-1, 2, size=(100, 100, 3)).astype(np.float32)
    result_image = image_array_to_image(img_array)
    assert isinstance(result_image, Image.Image)
    assert result_image.size == (100, 100)
    assert result_image.mode == "RGB"
    assert np.array(result_image).dtype == np.uint8
    assert np.array(result_image).min() >= 0 and np.array(result_image).max() <= 255


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
    writer = ImageWriter(write_dir=tmp_path)
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
    writer = ImageWriter(write_dir=tmp_path, num_processes=2, num_threads=2)
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
    writer = ImageWriter(write_dir=tmp_path)
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
    writer = ImageWriter(write_dir=tmp_path, num_processes=2, num_threads=2)
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
    writer = ImageWriter(write_dir=tmp_path)
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
    writer = ImageWriter(write_dir=tmp_path, num_processes=2, num_threads=2)
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
    writer = ImageWriter(write_dir=tmp_path)
    try:
        image_array = "invalid data"
        fpath = writer.get_image_file_path(0, "test_key", 0)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        with patch("builtins.print") as mock_print:
            writer.save_image(image_array, fpath)
            writer.wait_until_done()
            mock_print.assert_called()
            assert not fpath.exists()
    finally:
        writer.stop()


def test_save_image_after_stop(tmp_path, img_array_factory):
    writer = ImageWriter(write_dir=tmp_path)
    writer.stop()
    image_array = img_array_factory()
    fpath = writer.get_image_file_path(0, "test_key", 0)
    writer.save_image(image_array, fpath)
    time.sleep(1)
    assert not fpath.exists()


def test_stop(tmp_path):
    writer = ImageWriter(write_dir=tmp_path, num_processes=0, num_threads=2)
    writer.stop()
    assert not any(t.is_alive() for t in writer.threads)


def test_stop_multiprocessing(tmp_path):
    writer = ImageWriter(write_dir=tmp_path, num_processes=2, num_threads=2)
    writer.stop()
    assert not any(p.is_alive() for p in writer.processes)


def test_multiple_stops(tmp_path):
    writer = ImageWriter(write_dir=tmp_path)
    writer.stop()
    writer.stop()  # Should not raise an exception
    assert not any(t.is_alive() for t in writer.threads)


def test_multiple_stops_multiprocessing(tmp_path):
    writer = ImageWriter(write_dir=tmp_path, num_processes=2, num_threads=2)
    writer.stop()
    writer.stop()  # Should not raise an exception
    assert not any(t.is_alive() for t in writer.threads)


def test_wait_until_done(tmp_path, img_array_factory):
    writer = ImageWriter(write_dir=tmp_path, num_processes=0, num_threads=4)
    try:
        num_images = 100
        image_arrays = [img_array_factory(width=500, height=500) for _ in range(num_images)]
        fpaths = [writer.get_image_file_path(0, "test_key", i) for i in range(num_images)]
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
    writer = ImageWriter(write_dir=tmp_path, num_processes=2, num_threads=2)
    try:
        num_images = 100
        image_arrays = [img_array_factory() for _ in range(num_images)]
        fpaths = [writer.get_image_file_path(0, "test_key", i) for i in range(num_images)]
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
    writer = ImageWriter(write_dir=tmp_path)
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
    writer = ImageWriter(write_dir=tmp_path)
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
            self.image_writer = MagicMock(spec=ImageWriter)

    @safe_stop_image_writer
    def function_that_raises_exception(dataset=None):
        raise Exception("Test exception")

    dataset = MockDataset()

    with pytest.raises(Exception) as exc_info:
        function_that_raises_exception(dataset=dataset)

    assert str(exc_info.value) == "Test exception"
    dataset.image_writer.stop.assert_called_once()


def test_main_process_time(tmp_path, img_tensor_factory):
    writer = ImageWriter(write_dir=tmp_path)
    try:
        image_tensor = img_tensor_factory()
        fpath = tmp_path / "test_main_process_time.png"
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
