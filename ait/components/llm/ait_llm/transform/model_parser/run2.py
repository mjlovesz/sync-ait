from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM

from parser import model_to_json

model_to_json(Qwen2ForCausalLM(Qwen2Config()), "qwen2")
