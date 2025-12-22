import copy


from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from .modeling_siglip2 import Siglip2VisionConfig

logger = logging.get_logger(__name__)


class Eagle3_VLConfig(PretrainedConfig):
    model_type = 'eagle_3_vl'
    is_composition = True
    sub_configs = {"vision_config": SiglipVisionConfig, "text_config": Qwen2Config}
    def __init__(
            self,
            vision_config=None,
            text_config=None,
            use_backbone_lora=0,
            use_llm_lora=0,
            pad2square=False,
            select_layer=-4,
            downsample_ratio=0.5,
            template=None,
            loss_version='v1',
            mlp_checkpoint=False,
            image_token_index=151667,
            **kwargs):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {'model_type': 'siglip_vision_model'}
            logger.info('vision_config is None. Initializing the InternVisionConfig with default values.')

        if text_config is None:
            text_config = {'architectures': ['Qwen2ForCausalLM']}
            logger.info('text_config is None. Initializing the LlamaConfig config with default values (`LlamaConfig`).')

        if vision_config['model_type'] == 'siglip_vision_model':
            self.vision_config = SiglipVisionConfig(**vision_config)
        elif vision_config['model_type'] == 'siglip2_vision_model':
            self.vision_config = Siglip2VisionConfig(**vision_config)
        elif vision_config['model_type'] == 'intern_vit_6b':
            self.vision_config = InternVisionConfig(**vision_config)
        elif vision_config['model_type'] == 'radio':
            self.vision_config = RADIOConfig(**vision_config)
        else:
            raise ValueError('Unsupported model_type: {}'.format(vision_config['model_type']))


        if text_config['architectures'][0] == 'LlamaForCausalLM':
            self.text_config = LlamaConfig(**text_config)
        elif text_config['architectures'][0] == 'Phi3ForCausalLM':
            self.text_config = Phi3Config(**text_config)
        elif text_config['architectures'][0] == 'Qwen2ForCausalLM':
            self.text_config = Qwen2Config(**text_config)
        elif text_config['architectures'][0] == 'Qwen3ForCausalLM':
            self.text_config = Qwen3Config(**text_config)
        else:
            raise ValueError('Unsupported architecture: {}'.format(text_config['architectures'][0]))
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.mlp_checkpoint = mlp_checkpoint
        self.pad2square = pad2square
        self.select_layer = select_layer
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.loss_version = loss_version
        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        self.image_token_index = image_token_index

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['vision_config'] = self.vision_config.to_dict()
        output['text_config'] = self.text_config.to_dict()
        output['model_type'] = self.__class__.model_type
        output['use_backbone_lora'] = self.use_backbone_lora
        output['use_llm_lora'] = self.use_llm_lora
        output['select_layer'] = self.select_layer
        output['downsample_ratio'] = self.downsample_ratio
        output['template'] = self.template
        output['image_token_index'] = self.image_token_index
        output['_attn_implementation'] = self._attn_implementation
        output['_attn_implementation_autoset'] = self._attn_implementation_autoset
        return output
