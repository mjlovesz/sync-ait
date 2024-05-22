from transformers.models.bloom.modeling_bloom import BloomConfig,BloomForCausalLM

from parser import module_to_json

module_to_json(BloomForCausalLM(BloomConfig()), "bloom")
