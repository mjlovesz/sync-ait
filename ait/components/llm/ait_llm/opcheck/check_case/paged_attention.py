# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch_npu

import numpy as np

from ait_llm.opcheck import operation_test
from ait_llm.common.log import logger


class OpcheckPagedAttentionAttentionOperation(operation_test.OperationTest):
    def group_matmul(head, kv_head, A, B):
        group_num = head // kv_head
        score = None
        for i in range(kv_head):
            group_score = np.matmul(A[i * group_num: (i + 1) * group_num, :, :].astype(np.float32), 
                                    B[i: (i + 1), :, :].astype(np.float32))
            if score is None:
                score = group_score
            else:
                score = np.concatenate((score, group_score), 0)
        logger.debug(score.shape)
        return score                                                                                                                                                                                                                                                                                                                                                                                                                     

    def ref_masked_attention(
            query, # (1, num_heads, head_size)
            key, # (context_len, kv_heads, head_size)
            value,
            scale: float,
            alibi_bias=None
    ):
        # Q * K.T
        query = query * scale
        query = np.transpose(query, (1, 0, 2))
        key = np.transpose(key, (1, 2, 0))
        sim = OpcheckPagedAttentionAttentionOperation.group_matmul(query.shape[0], key.shape[0], query, key)
        if alibi_bias is not None:
            sim = sim + alibi_bias
        # softmax
        row_max = np.max(sim, axis=-1, keepdims=True)
        sim -= row_max
        sim = sim.astype(np.float32)
        sim = np.exp(sim)
        row_sum = np.sum(sim, axis=-1, keepdims=True)
        p = sim / row_sum
        p = p.astype(np.float16)
        # P * V
        value = np.transpose(value, (1, 0, 2))
        out = OpcheckPagedAttentionAttentionOperation.group_matmul(query.shape[0], key.shape[0], p, value)
        out = np.transpose(out, (1, 0, 2))
        return out

    def ref_single_query_cached_kv_attention(output, paged_input, alibi=None) -> None:
        query = paged_input[0]
        key_cache = paged_input[1]
        value_cache = paged_input[2]
        block_tables = paged_input[3]
        context_lens = paged_input[4]
        num_heads = query.shape[1]
        kv_heads = value_cache.shape[2]
        head_size = value_cache.shape[3]
        block_size = value_cache.shape[1]
        num_input_tokens = query.shape[0]
        for i in range(num_input_tokens):
            q = np.expand_dims(query[i], 0)
            block_table = block_tables[i]
            context_len = int(context_lens[i])
            keys = []
            values = []
            for j in range(context_len):
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size
                k = key_cache[block_number, block_offset, :, :]
                k = k.reshape(kv_heads, head_size)
                keys.append(k)
                v = value_cache[block_number, block_offset, :, :]
                v = v.reshape(kv_heads, head_size)
                values.append(v)
            keys = np.stack(np.array(values), axis=0)
            values = np.stack(np.array(values), axis=0)
            scale = 1.0 / (head_size ** 0.5)
            if alibi is None:
                out = OpcheckPagedAttentionAttentionOperation.ref_masked_attention(q, keys, values, scale)
            else:
                out = OpcheckPagedAttentionAttentionOperation.ref_masked_attention(q, keys, values, scale, alibi[i, :, :, :context_len])
            out = out.reshape(num_heads, head_size)
            output[i] = out

    def nz_2_nd(self, data):
        origin_shape = data.shape
        dims = list(range(len(origin_shape)))
        last_dims = dims[-4:]

        perm = dims[:-4] + [last_dims[-1], last_dims[2], last_dims[0], last_dims[3]]
        data = data.transpose(perm)
        nd_shape = data.shape[:-4] + (data.shape[-4] * data.shape[-3], data.shape[-2], data.shape[-1])
        data = data.reshape(nd_shape)
        return data

    def golden_calc(self, in_tensors):
        alibi_mask = None
        soc_version = self.get_soc_version()
        is_support_alibi = self.op_param.get("isSupportAlibi", False)
        if is_support_alibi and soc_version == "Ascend310P":
            query, key_cache_nz, value_cache_nz, block_tables, context_lens, alibi_mask_nz = in_tensors
        elif not is_support_alibi and soc_version == "Ascend310P":
            query, key_cache_nz, value_cache_nz, block_tables, context_lens = in_tensors
        elif is_support_alibi and soc_version == "Ascend910B":
            query, key_cache, value_cache, block_tables, context_lens, alibi_mask = in_tensors
        else:
            query, key_cache, value_cache, block_tables, context_lens = in_tensors

        if soc_version == "Ascend310P":
            key_cache = self.nz_2_nd(key_cache_nz)
            value_cache = self.nz_2_nd(value_cache_nz)
            if is_support_alibi:
                alibi_mask = self.nz_2_nd(alibi_mask_nz)

        ref_output = torch.zeros_like(query)
        input = query, key_cache, value_cache, block_tables, context_lens
        OpcheckPagedAttentionAttentionOperation.ref_single_query_cached_kv_attention(
            ref_output,
            input,
            alibi_mask
        )

    def test(self):
        self.execute()