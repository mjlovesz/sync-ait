from transformers.models.bloom.modeling_bloom import BloomConfig, BloomForCausalLM

from parser import model_to_json

model = BloomForCausalLM(BloomConfig())

model_to_json(model, "bloom")
