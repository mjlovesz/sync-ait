from transformers.models.starcoder2.modeling_starcoder2 import Starcoder2Config, Starcoder2ForCausalLM

from ..parser import model_to_json

model_to_json(Starcoder2ForCausalLM(Starcoder2Config()), "starcoder2")
