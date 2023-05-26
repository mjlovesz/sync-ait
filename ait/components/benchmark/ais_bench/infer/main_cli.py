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

import click

from ais_bench.infer.benchmark_process import benchmark_process
from ais_bench.infer.args_adapter import BenchMarkArgsAdapter
from ais_bench.infer.options import (
    opt_model,
    opt_input_path,
    opt_output,
    opt_output_dirname,
    opt_outfmt,
    opt_loop,
    opt_debug,
    opt_device,
    opt_dym_batch,
    opt_dym_hw,
    opt_dym_dims,
    opt_dym_shape,
    opt_output_size,
    opt_auto_set_dymshape_mode,
    opt_auto_set_dymdims_mode,
    opt_batchsize,
    opt_pure_data_type,
    opt_profiler,
    opt_dump,
    opt_acl_json_path,
    opt_output_batchsize_axis,
    opt_run_mode,
    opt_display_all_summary,
    opt_warmup_count,
    opt_dym_shape_range,
    opt_aipp_config,
    opt_energy_consumption,
    opt_npu_id
)


@click.command(name="benchmark", 
               short_help = "benchmark tool to get performance data including latency and throughput",
               no_args_is_help=True)
@opt_model
@opt_input_path
@opt_output
@opt_output_dirname
@opt_outfmt
@opt_loop
@opt_debug
@opt_device
@opt_dym_batch
@opt_dym_hw
@opt_dym_dims
@opt_dym_shape
@opt_output_size
@opt_auto_set_dymshape_mode
@opt_auto_set_dymdims_mode
@opt_batchsize
@opt_pure_data_type
@opt_profiler
@opt_dump
@opt_acl_json_path
@opt_output_batchsize_axis
@opt_run_mode
@opt_display_all_summary
@opt_warmup_count
@opt_dym_shape_range
@opt_aipp_config
@opt_energy_consumption
@opt_npu_id
@opt_backend
@opt_perf
def benchmark_cli(om_model, input_path, output, 
                  output_dirname, outfmt, loop, debug, device,
                  dym_batch, dym_hw, dym_dims, dym_shape, output_size, auto_set_dymshape_mode,
                  auto_set_dymdims_mode, batch_size, pure_data_type, profiler, dump,
                  acl_json_path, output_batchsize_axis, run_mode, display_all_summary,
                  warmup_count, dym_shape_range, aipp_config, energy_consumption, npu_id, backend, perf):

    args = BenchMarkArgsAdapter(om_model.as_posix(), input_path, output.as_posix() if output else None, 
                                output_dirname, outfmt, loop, debug, device,
                                dym_batch, dym_hw, dym_dims, dym_shape, output_size, auto_set_dymshape_mode,
                                auto_set_dymdims_mode, batch_size, pure_data_type, profiler, dump,
                                acl_json_path, output_batchsize_axis, run_mode, display_all_summary,
                                warmup_count, dym_shape_range, aipp_config, energy_consumption, npu_id, backend, perf)
    benchmark_process(args)