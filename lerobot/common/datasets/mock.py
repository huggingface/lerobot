import unittest

from lerobot_dataset import MultiLeRobotDataset


class TestMultiLeRobotDataset(unittest.TestCase):
    def setUp(self):
        # Define the datasets to use
        self.dataset_repo_ids = ["lerobot/aloha_sim_insertion_human", "lerobot/aloha_static_vinh_cup"]
        self.dataset = MultiLeRobotDataset(
            repo_ids=self.dataset_repo_ids,
            # Replace with your local path or None for Hugging Face Hub
            split="train",
            image_transforms=None,  # Pass your transforms if any
            delta_timestamps=None,
        )

    def test_initialization(self):
        # Check if datasets were initialized correctly
        self.assertEqual(len(self.dataset.repo_ids), 2)
        self.assertEqual(self.dataset.repo_ids, self.dataset_repo_ids)

    def test_num_samples(self):
        # Check the total number of samples
        self.assertGreater(len(self.dataset), 0)

    def test_num_episodes(self):
        # Check the total number of episodes
        self.assertGreater(self.dataset.num_episodes, 0)

    def test_fps(self):
        # Check that FPS is correctly returned and is consistent
        fps = self.dataset.fps
        self.assertGreater(fps, 0)

    def test_video_property(self):
        # Check if video loading is correctly handled
        self.assertIsInstance(self.dataset.video, bool)

    def test_getitem(self):
        # Test accessing a few samples to see if they are returned correctly
        for i in range(5):
            sample = self.dataset[i]
            self.assertIsInstance(sample, dict)
            self.assertIn("dataset_index", sample)  # Check that dataset index is included
            breakpoint()

    def test_camera_keys(self):
        # Test that camera keys are returned correctly
        camera_keys = self.dataset.camera_keys
        self.assertIsInstance(camera_keys, list)
        self.assertGreater(len(camera_keys), 0)

    def test_video_frame_keys(self):
        # Test that video frame keys are returned correctly
        video_frame_keys = self.dataset.video_frame_keys
        if self.dataset.video:
            self.assertIsInstance(video_frame_keys, list)
        else:
            self.assertEqual(len(video_frame_keys), 0)


if __name__ == "__main__":
    unittest.main()
