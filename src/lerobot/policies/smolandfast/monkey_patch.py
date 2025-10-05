from transformers.models.smolvlm.processing_smolvlm import SmolVLMProcessor


def patch_SmolVLMProcessor():  # noqa: N802
    SmolVLMProcessor.image_processor_class = "SmolVLMImageProcessorFast"
