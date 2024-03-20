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

import sys
import os
import unittest
import json
import math
import torch
import torch_npu
import numpy as np

from ait_llm.opcheck import operation_test
from ait_llm.common.log import logger


class OpcheckUnpadSelfAttentionOperation(operation_test.OperationTest):
    def group_matmul(self, heads, group_num, in_a, in_b):
        try:
            group_head = heads // group_num
        except ZeroDivisionError as e:
            raise e       
        score = None
        for i in range(group_num):
            group_score = np.matmul(in_a[i * group_head: (i + 1) * group_head, :, :].astype(np.float32),
                                    in_b[i:(i + 1), :, :].astype(np.float32)).astype(np.float16)
            if score is None:
                score = group_score
            else:
                score = np.concatenate((score, group_score), 0)
        logger.debug(score.shape)
        return score

    def encoder_golden_func(self, in_tensors):
        mixed_q, mixed_k, mixed_v, attention_mask, seq_len = in_tensors[0], in_tensors[1], in_tensors[2], \
            in_tensors[3], in_tensors[4]
        
        if self.op_param["batchRunStatusEnable"]:
            batch_status = in_tensors[5]
        else:
            batch_status = len(seq_len)

        heads, group_num, embed = self.op_param["headNum"], self.op_param["kvHeadNum"], 128
        q_seqlen = kv_seqlen = seq_len # crossattention时，q_seqlen != k_seqlen 
        max_s, ntokens2 = np.max(q_seqlen), (q_seqlen * kv_seqlen).sum()

        q_offset, k_offset, v_offset = 0, 0, 0
        s, _p, out = None, None, None

        for idx, _ in enumerate(range(batch_status)):
            q_s, kv_s = q_seqlen[idx], kv_seqlen[idx]
            q_slice = mixed_q[q_offset:q_offset + q_s][:].reshape(q_s, heads, embed)
            q_slice = np.transpose(q_slice, (1, 0, 2))  # (heads, q_seq, embed)
            k_slice = mixed_k[k_offset:k_offset + kv_s][:].reshape(kv_s, group_num, embed)
            k_slice = np.transpose(k_slice, (1, 0, 2))
            k_slice_t = np.transpose(k_slice, (0, 2, 1))   # get K^T (kv_heads, embed, k_seq)
            v_slice = mixed_v[v_offset:v_offset + kv_s][:].reshape(kv_s, group_num, embed)
            v_slice = np.transpose(v_slice, (1, 0, 2))
            score = self.group_matmul(heads, group_num, q_slice, k_slice_t)
            s = score.reshape([-1, ]) if s is None else np.concatenate((s, score.reshape([-1, ])), 0)
            
            try:
                score = score * np.float16(1.0 / math.sqrt(1.0 * embed))
            except ZeroDivisionError as e:
                raise e

            score = score + attention_mask[:, :q_s, :kv_s] if self.op_param["isTriuMask"] else score
            score_max = np.max(score, axis=-1)
            score = score - score_max.reshape((heads, q_s, 1))
            score_exp = np.exp(score.astype(np.float32))
            if not self.op_param["isFp32"]:
                score_sum = np.sum(score_exp.astype(np.float16), axis=-1)
                _p = score_exp.astype(np.float16).reshape([-1, ]) if _p is None else \
                    np.concatenate((_p, score_exp.astype(np.float16).reshape([-1, ])), 0)
                try:
                    p = score_exp.astype(np.float16) / score_sum.reshape((heads, q_s, 1)).astype(np.float16)
                except ZeroDivisionError as e:
                    raise e
                out_sub = self.group_matmul(heads, group_num, p, v_slice)
            else:
                score_sum = np.sum(score_exp, axis=-1)
                _p = score_exp.astype(np.float16).reshape([-1, ]) if _p is None else \
                    np.concatenate((_p, score_exp.astype(np.float16).reshape([-1, ])), 0)
                p = score_exp.astype(np.float16)
                out_sub = self.group_matmul(heads, group_num, p, v_slice)
                try:
                    out_sub = out_sub / score_sum.reshape((heads, q_s, 1)).astype(np.float16)
                except ZeroDivisionError as e:
                    raise e
            out_sub = out_sub.reshape(heads, q_s, embed)
            out_sub = np.transpose(out_sub, (1, 0, 2))
            out_sub = np.ascontiguousarray(out_sub)
            out = out_sub if out is None else np.concatenate((out, out_sub), 0)

        return out.astype("float16").reshape(-1, heads, 128)

    def not_encoder_golden_func(self, in_tensors):
        mixed_q, mixed_k, mixed_v, cache_k, cache_v, attention_mask, token_offset, seq_len, layerid = in_tensors[0], \
            in_tensors[1], in_tensors[2], in_tensors[3], in_tensors[4], in_tensors[5], in_tensors[6], in_tensors[7], \
            int(in_tensors[8][0])
        if self.op_param["batchRunStatusEnable"]:
            batch_status = in_tensors[9]
        else:
            batch_status = len(seq_len)
        q_scale, qk_scale, head_num, head_size = self.op_param["qScale"], self.op_param["qkScale"], \
            self.op_param["headNum"], self.op_param["headDim"]
        offset = 0
        context_list = []

        for i, _ in enumerate(range(batch_status)):
            cur_seqlen = seq_len[i]
            cur_token_offset = token_offset[i]
            cur_token_offset_start = cur_token_offset - cur_seqlen
            next_offset = offset + cur_seqlen
            cur_q = mixed_q[offset:next_offset]
            cur_k = mixed_k[offset:next_offset]
            cur_v = mixed_v[offset:next_offset]
            if cur_token_offset_start > 0:
                past_k = cache_k[layerid, i, :cur_token_offset_start, :]
                past_v = cache_v[layerid, i, :cur_token_offset_start, :]
                cur_k = torch.concat([past_k, cur_k], dim=0)
                cur_v = torch.concat([past_v, cur_v], dim=0)
            cur_q = (cur_q * q_scale).view(cur_seqlen, head_num, head_size).transpose(0, 1)
            cur_k = cur_k.view(cur_token_offset, head_num, head_size).permute(1, 2, 0)
            cur_qk = torch.bmm(cur_q, cur_k) # [head_num, seqlen, token_offset]
            if self.op_param["isClamp"]:
                clamp_min = self.op_param["clampMin"]
                clamp_max = self.op_param["clampMax"]
                cur_qk = torch.clamp(cur_qk, clamp_min, clamp_max)
            if attention_mask.ndim == 3: # masked_fill
                cur_qk = cur_qk + attention_mask[i, :cur_seqlen, :cur_token_offset]
            else:
                cur_qk = cur_qk + attention_mask[:cur_seqlen, :cur_token_offset]
            cur_qk = cur_qk * qk_scale
            cur_qk = torch.nn.functional.softmax(cur_qk.type(torch.float32), dim=-1).type(torch.float16)

            cur_v = cur_v.view(cur_token_offset, head_num, head_size).transpose(0, 1)
            cur_context = torch.bmm(cur_qk, cur_v).transpose(0, 1).contiguous().view(cur_seqlen, head_num * head_size)
            context_list.append(cur_context)

            offset = next_offset

        out = torch.concat(context_list, dim=0)
        return out

    def golden_calc(self, in_tensors):
        if self.op_param["isEncoder"]:
            out = self.encoder_golden_func(in_tensors)
        else:
            out = self.not_encoder_golden_func(in_tensors)
        return [out]

    def test(self):
        soc_version = self.get_soc_version()
        if soc_version != 'Ascend910B':
            logger_text = "{} is not supported! Only supports Ascend910B!".format(soc_version)
            logger.error(logger_text)
            return

        if self.op_param["isEncoder"]:
            if self.op_param["batchRunStatusEnable"]:
                self.case_info["run_param"] = json.dumps({"seqLen": self.in_tensors[4].tolist(),
                                                        "batchRunStatus": self.in_tensors[5].tolist()})
            else:
                self.case_info["run_param"] = json.dumps({"seqLen": self.in_tensors[4].tolist()})
        else:
            if self.op_param["batchRunStatusEnable"]:
                self.case_info['run_param'] = json.dumps({"tokenOffset": self.in_tensors[6].tolist(), 
                                                        "seqLen": self.in_tensors[7].tolist(),
                                                        "batchRunStatus": self.in_tensors[9].tolist()})
            else:
                self.case_info["run_param"] = json.dumps({"tokenOffset": self.in_tensors[6].tolist(), 
                                                        "seqLen": self.in_tensors[7].tolist()})

        self.execute_with_param()