import torch

from lerobot.common.policies.reward_model.configuration_classifier import RewardClassifierConfig
from lerobot.common.policies.reward_model.modeling_classifier import ClassifierOutput
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from tests.utils import require_package


def test_classifier_output():
    output = ClassifierOutput(
        logits=torch.tensor([1, 2, 3]),
        probabilities=torch.tensor([0.1, 0.2, 0.3]),
        hidden_states=None,
    )

    assert (
        f"{output}"
        == "ClassifierOutput(logits=tensor([1, 2, 3]), probabilities=tensor([0.1000, 0.2000, 0.3000]), hidden_states=None)"
    )


@require_package("transformers")
def test_binary_classifier_with_default_params():
    from lerobot.common.policies.reward_model.modeling_classifier import Classifier

    config = RewardClassifierConfig()
    config.input_features = {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        "next.reward": PolicyFeature(type=FeatureType.REWARD, shape=(1,)),
    }
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "REWARD": NormalizationMode.IDENTITY,
    }
    config.num_cameras = 1
    classifier = Classifier(config)

    batch_size = 10

    input = {
        "observation.image": torch.rand((batch_size, 3, 224, 224)),
        "next.reward": torch.randint(low=0, high=2, size=(batch_size,)).float(),
    }

    images, labels = classifier.extract_images_and_labels(input)
    assert len(images) == 1
    assert images[0].shape == torch.Size([batch_size, 3, 224, 224])
    assert labels.shape == torch.Size([batch_size])

    output = classifier.predict(images)

    assert output is not None
    assert output.logits.size() == torch.Size([batch_size])
    assert not torch.isnan(output.logits).any(), "Tensor contains NaN values"
    assert output.probabilities.shape == torch.Size([batch_size])
    assert not torch.isnan(output.probabilities).any(), "Tensor contains NaN values"
    assert output.hidden_states.shape == torch.Size([batch_size, 512])
    assert not torch.isnan(output.hidden_states).any(), "Tensor contains NaN values"


@require_package("transformers")
def test_multiclass_classifier():
    from lerobot.common.policies.reward_model.modeling_classifier import Classifier

    num_classes = 5
    config = RewardClassifierConfig()
    config.input_features = {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        "next.reward": PolicyFeature(type=FeatureType.REWARD, shape=(num_classes,)),
    }
    config.num_cameras = 1
    config.num_classes = num_classes
    classifier = Classifier(config)

    batch_size = 10

    input = {
        "observation.image": torch.rand((batch_size, 3, 224, 224)),
        "next.reward": torch.rand((batch_size, num_classes)),
    }

    images, labels = classifier.extract_images_and_labels(input)
    assert len(images) == 1
    assert images[0].shape == torch.Size([batch_size, 3, 224, 224])
    assert labels.shape == torch.Size([batch_size, num_classes])

    output = classifier.predict(images)

    assert output is not None
    assert output.logits.shape == torch.Size([batch_size, num_classes])
    assert not torch.isnan(output.logits).any(), "Tensor contains NaN values"
    assert output.probabilities.shape == torch.Size([batch_size, num_classes])
    assert not torch.isnan(output.probabilities).any(), "Tensor contains NaN values"
    assert output.hidden_states.shape == torch.Size([batch_size, 512])
    assert not torch.isnan(output.hidden_states).any(), "Tensor contains NaN values"


@require_package("transformers")
def test_default_device():
    from lerobot.common.policies.reward_model.modeling_classifier import Classifier

    config = RewardClassifierConfig()
    assert config.device == "cpu"

    classifier = Classifier(config)
    for p in classifier.parameters():
        assert p.device == torch.device("cpu")


@require_package("transformers")
def test_explicit_device_setup():
    from lerobot.common.policies.reward_model.modeling_classifier import Classifier

    config = RewardClassifierConfig(device="cpu")
    assert config.device == "cpu"

    classifier = Classifier(config)
    for p in classifier.parameters():
        assert p.device == torch.device("cpu")
