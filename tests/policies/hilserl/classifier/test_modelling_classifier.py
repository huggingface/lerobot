import torch

from lerobot.common.policies.hilserl.classifier.modeling_classifier import (
    ClassifierConfig,
    ClassifierOutput,
)
from tests.utils import require_package


def test_classifier_output():
    output = ClassifierOutput(
        logits=torch.tensor([1, 2, 3]), probabilities=torch.tensor([0.1, 0.2, 0.3]), hidden_states=None
    )

    assert (
        f"{output}"
        == "ClassifierOutput(logits=tensor([1, 2, 3]), probabilities=tensor([0.1000, 0.2000, 0.3000]), hidden_states=None)"
    )


@require_package("transformers")
def test_binary_classifier_with_default_params():
    from lerobot.common.policies.hilserl.classifier.modeling_classifier import Classifier

    config = ClassifierConfig()
    classifier = Classifier(config)

    batch_size = 10

    input = torch.rand(batch_size, 3, 224, 224)
    output = classifier(input)

    assert output is not None
    assert output.logits.shape == torch.Size([batch_size])
    assert not torch.isnan(output.logits).any(), "Tensor contains NaN values"
    assert output.probabilities.shape == torch.Size([batch_size])
    assert not torch.isnan(output.probabilities).any(), "Tensor contains NaN values"
    assert output.hidden_states.shape == torch.Size([batch_size, 2048])
    assert not torch.isnan(output.hidden_states).any(), "Tensor contains NaN values"


@require_package("transformers")
def test_multiclass_classifier():
    from lerobot.common.policies.hilserl.classifier.modeling_classifier import Classifier

    num_classes = 5
    config = ClassifierConfig(num_classes=num_classes)
    classifier = Classifier(config)

    batch_size = 10

    input = torch.rand(batch_size, 3, 224, 224)
    output = classifier(input)

    assert output is not None
    assert output.logits.shape == torch.Size([batch_size, num_classes])
    assert not torch.isnan(output.logits).any(), "Tensor contains NaN values"
    assert output.probabilities.shape == torch.Size([batch_size, num_classes])
    assert not torch.isnan(output.probabilities).any(), "Tensor contains NaN values"
    assert output.hidden_states.shape == torch.Size([batch_size, 2048])
    assert not torch.isnan(output.hidden_states).any(), "Tensor contains NaN values"


@require_package("transformers")
def test_default_device():
    from lerobot.common.policies.hilserl.classifier.modeling_classifier import Classifier

    config = ClassifierConfig()
    assert config.device == "cpu"

    classifier = Classifier(config)
    for p in classifier.parameters():
        assert p.device == torch.device("cpu")


@require_package("transformers")
def test_explicit_device_setup():
    from lerobot.common.policies.hilserl.classifier.modeling_classifier import Classifier

    config = ClassifierConfig(device="meta")
    assert config.device == "meta"

    classifier = Classifier(config)
    for p in classifier.parameters():
        assert p.device == torch.device("meta")
