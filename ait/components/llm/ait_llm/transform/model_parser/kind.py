from typing import List

from torch.nn import Module, Linear, Embedding, LayerNorm, GELU


def convert(module: Module):
    module_class = module.__class__
    module_name = module_class.__name__
    lowered = module_name.lower()

    if "rms" in lowered:
        return rms_norm(module)
    elif module_class is Linear:
        return linear(module)
    elif module_class is Embedding:
        return embedding(module)
    elif module_class is LayerNorm:
        return layernorm(module)
    elif "rotary" in lowered and "embedding" in lowered:
        return rope(module)
    else:
        return {"children": []}


def linear(module):
    return {
        "kind": "Linear",
        "in_features": module.in_features,
        "out_features": module.out_features,
        "bias": module.bias is not None
    }


def embedding(module):
    return {
        "kind": "Embedding",
        "num_embeddings": module.num_embeddings,
        "embedding_dim": module.embedding_dim,
        "padding_idx": module.padding_idx
    }


def attention(modules: List[Module], size: int):
    ret = {}

    if size == 5:
        [q, k, v, o, r] = modules
        ret["structure"] = "q-k-v-o-r"
        ret["q"] = linear(q)
        ret["k"] = linear(k)
        ret["v"] = linear(v)
        ret["o"] = linear(o)
        ret["rope"] = rope(r)
    elif size == 4:
        [q, kv, o, r] = modules
        ret["structure"] = "q-kv-o-r"
        ret["q"] = linear(q)
        ret["kv"] = linear(kv)
        ret["o"] = linear(o)
        ret["rope"] = rope(r)
    elif size == 3:
        [w, o, r] = modules
        ret["structure"] = "w-o-r"
        ret["w"] = linear(w)
        ret["o"] = linear(o)
        ret["rope"] = rope(r)
    elif size == 2:
        [w, o] = modules
        ret["structure"] = "w-o"
        ret["w"] = linear(w)
        ret["o"] = linear(o)
    else:
        print("error linear size")

    return ret


def mlp(modules: List[Module]):
    ret = {"ff": []}

    for m in modules:
        if isinstance(m, Linear):
            ret["ff"].append(linear(m))
        else:
            ret["act"] = activation(m)

    return ret


def layernorm(module):
    return {
        "kind": "LayerNorm",
        "normalized_shape": str(module.normalized_shape),
        "eps": module.eps,
        "element_affine": module.elementwise_affine,
        "bias": module.bias is not None
    }


def rope(module: Module):
    return {
        "kind": "RotaryEmbedding",
        "base": module.base if hasattr(module, "base") else -1,
        "dim": module.dim if hasattr(module, "dim") else -1,
        "max_position_embeddings": module.max_position_embeddings if hasattr(module, "max_position_embeddings") else -1,
        "max_seq_len_cached": module.max_seq_len_cached if hasattr(module, "max_seq_len_cached") else -1
    }


def rms_norm(module: Module):
    ret = {"kind": "RMSNorm"}
    eps_like = ["epsilon", "variance_epsilon", "eps"]

    for name in eps_like:
        if hasattr(module, name):
            ret["eps"] = getattr(module, name)
            break

    return ret


def activation(module: Module):
    module_class = module.__class__

    if module_class is GELU:
        return {
            "kind": "GELU",
            "approximate": module.approximate
        }

    module_name = module_class.__name__
    lowered = module_name.lower()

    if "tanh" in lowered and "gelu" in lowered:
        return {
            "kind": "GELU",
            "approximate": True
        }

    return {"kind": module_name}
