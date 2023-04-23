# Copyright 2023 Huawei Technologies Co., Ltd
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

from ais_bench.infer.main_enter import main_enter
from ais_bench.infer.args_adapter import MyArgs
from ais_bench.infer.options import (
    opt_model,
    opt_input,
    opt_output,
    opt_output_dirname,
    opt_outfmt,
    opt_loop,
    opt_debug,
    opt_device,
    opt_dymBatch,
    opt_dymHW,
    opt_dymDims,
    opt_dymShape,
    opt_outputSize,
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
    opt_dymShape_range
)


@click.command(name="benchmark", short_help = "Inference tool to get performance data including latency and throughput")
@opt_model
@opt_input
@opt_output
@opt_output_dirname
@opt_outfmt
@opt_loop
@opt_debug
@opt_device
@opt_dymBatch
@opt_dymHW
@opt_dymDims
@opt_dymShape
@opt_outputSize
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
@opt_dymShape_range
def benchmark_cli_enter(model, input, output, output_dirname, outfmt, loop, debug, device,
                        dymBatch, dymHW, dymDims, dymShape, outputSize, auto_set_dymshape_mode,
                        auto_set_dymdims_mode, batchsize, pure_data_type, profiler, dump,
                        acl_json_path, output_batchsize_axis, run_mode, display_all_summary,
                        warmup_count, dymShape_range):
    args = MyArgs(model.as_posix(), input.as_posix() if input else None, output.as_posix() if output else None,
                  output_dirname, outfmt, loop, debug, device,
                  dymBatch, dymHW, dymDims, dymShape, outputSize, auto_set_dymshape_mode,
                  auto_set_dymdims_mode, batchsize, pure_data_type, profiler, dump,
                  acl_json_path, output_batchsize_axis, run_mode, display_all_summary,
                  warmup_count, dymShape_range)
    main_enter(args)