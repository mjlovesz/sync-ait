from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM

from ..parser import model_to_json

model_to_json(LlamaForCausalLM(LlamaConfig()), "llama")
