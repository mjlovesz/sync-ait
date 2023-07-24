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

class BenchMarkArgsAdapter():
    def __init__(self, model, input_path, output, output_dirname, outfmt, loop,
                 debug, device, dym_batch, dym_hw, dym_dims,
                 dym_shape, output_size, auto_set_dymshape_mode,
                 auto_set_dymdims_mode, batchsize, pure_data_type,
                 profiler, dump, acl_json_path, output_batchsize_axis,
                 run_mode, display_all_summary, warmup_count, dym_shape_range, aipp_config,
                 energy_consumption, npu_id, backend, perf, pipeline, profiler_rename, dump_npy):
        self.model = model
        self.input = input_path
        self.output = output
        self.output_dirname = output_dirname
        self.outfmt = outfmt
        self.loop = loop
        self.debug = debug
        self.device = device
        self.dym_batch = dym_batch
        self.dym_hw = dym_hw
        self.dym_dims = dym_dims
        self.dym_shape = dym_shape
        self.output_size = output_size
        self.auto_set_dymshape_mode = auto_set_dymshape_mode
        self.auto_set_dymdims_mode = auto_set_dymdims_mode
        self.batchsize = batchsize
        self.pure_data_type = pure_data_type
        self.profiler = profiler
        self.dump = dump
        self.acl_json_path = acl_json_path
        self.output_batchsize_axis = output_batchsize_axis
        self.run_mode = run_mode
        self.display_all_summary = display_all_summary
        self.warmup_count = warmup_count
        self.dym_shape_range = dym_shape_range
        self.aipp_config = aipp_config
        self.energy_consumption = energy_consumption
        self.npu_id = npu_id
        self.backend = backend
        self.perf = perf
        self.pipeline = pipeline
        self.profiler_rename = profiler_rename
        self.dump_npy = dump_npy