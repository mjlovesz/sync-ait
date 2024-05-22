from transformers.models.gemma.modeling_gemma import GemmaConfig, GemmaForCausalLM

from ..parser import model_to_json

model_to_json(GemmaForCausalLM(GemmaConfig()), "gemma")
