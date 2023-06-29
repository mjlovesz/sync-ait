import click
import os
import logging
import sys
from click.testing import CliRunner
from ais_bench.infer.benchmark_process import args_rules
from ais_bench.infer.args_adapter import BenchMarkArgsAdapter
from ais_bench.infer.main_cli import benchmark_cli
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
    opt_npu_id,
    opt_backend,
    opt_perf,
    opt_pipeline
)

logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = '[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

@click.command(name="benchmark_all",
               short_help = "benchmark tool to get performance data including latency and throughput",
               no_args_is_help=True)
@opt_model
# @opt_input_path
# @opt_output
# @opt_output_dirname
# @opt_outfmt
# @opt_loop
# @opt_debug
# @opt_device
# @opt_dym_batch
# @opt_dym_hw
# @opt_dym_dims
# @opt_dym_shape
# @opt_output_size
# @opt_auto_set_dymshape_mode
# @opt_auto_set_dymdims_mode
# @opt_batchsize
# @opt_pure_data_type
# @opt_profiler
# @opt_dump
# @opt_acl_json_path
# @opt_output_batchsize_axis
# @opt_run_mode
# @opt_display_all_summary
# @opt_warmup_count
# @opt_dym_shape_range
# @opt_aipp_config
# @opt_energy_consumption
# @opt_npu_id
# @opt_backend
# @opt_perf
# @opt_pipeline
def benchmark_all_cmd(om_model#,
                    #  input_path,
                    #  output,
                    #  output_dirname,
                    #  outfmt,
                    #  loop,
                    #  debug,
                    #  device,
                    #  dym_batch,
                    #  dym_hw,
                    #  dym_dims,
                    #  dym_shape,
                    #  output_size,
                    #  auto_set_dymshape_mode,
                    #  auto_set_dymdims_mode,
                    #  batch_size,
                    #  pure_data_type,
                    #  profiler,
                    #  dump,
                    #  acl_json_path,
                    #  output_batchsize_axis,
                    #  run_mode,
                    #  display_all_summary,
                    #  warmup_count,
                    #  dym_shape_range,
                    #  aipp_config,
                    #  energy_consumption,
                    #  npu_id,
                    #  backend,
                    #  perf,
                    #  pipeline
                     ):
     click.echo('Hello %s!' % om_model)

class TestClass:
    @classmethod
    def setup_class(cls):
        """
        class level setup_class
        """
        cls.init(TestClass)

    def init(self):
        self.all_args = BenchMarkArgsAdapter (
            model=os.path.realpath("../testdata/resnet50/model/pth_resnet50_bs4.om"),
            input_path="datasets/",
            output="output/",
            output_dirname="outdir/",
            outfmt="NPY",
            loop=100,
            debug="0",
            device="0,1",
            dym_batch=16,
            dym_hw="224,224",
            dym_dims="1,3,224,224",
            dym_shape="1,3,224,224",
            output_size="10000",
            auto_set_dymshape_mode="0",
            auto_set_dymdims_mode="0",
            batchsize=16,
            pure_data_type="zero",
            profiler="0",
            dump="0",
            acl_json_path="acl.json",
            output_batchsize_axis=1,
            run_mode="array",
            display_all_summary="0",
            warmup_count=1,
            dym_shape_range="1~3,3,224,224-226",
            aipp_config="aipp.config",
            energy_consumption="0",
            npu_id="0",
            backend="trtexec",
            perf="0",
            pipeline="0",
        )

    def test_check_all_full_args_legality(self):
        runner = CliRunner()
        cmd_list = ["--om-model", self.all_args.model,
                    # "--input", self.all_args.input,
                    # "--output", self.all_args.output,
                    # "--output-dirname", self.all_args.output_dirname,
                    # "--outfmt", self.all_args.outfmt,
                    # "--loop", self.all_args.loop,
                    # "--debug", self.all_args.debug,
                    # "--device", self.all_args.device,
                    # "--dym-batch", self.all_args.dym_batch,
                    # "--dym-hw", self.all_args.dym_hw,
                    # "--dym-dims", self.all_args.dym_dims,
                    # "--dym-shape", self.all_args.dym_shape,
                    # "--dym-shape-range", self.all_args.output_size,
                    # "--output-size", self.all_args.model,
                    # "--auto-set-dymshape-mode", self.all_args.auto_set_dymshape_mode,
                    # "--auto-set-dymdims-mode", self.all_args.auto_set_dymdims_mode,
                    # "--batch-size", self.all_args.batchsize,
                    # "--pure-data-type", self.all_args.pure_data_type,
                    # "--profiler", self.all_args.profiler,
                    # "--dump", self.all_args.dump,
                    # "--acl-json-path", self.all_args.acl_json_path,
                    # "--output-batchsize-axis", self.all_args.output_batchsize_axis,
                    # "--run-mode", self.all_args.run_mode,
                    # "--display-all-summary", self.all_args.display_all_summary,
                    # "--warmup-count", self.all_args.warmup_count,
                    # "--dym-shape-range", self.all_args.dym_shape_range,
                    # "--aipp-config", self.all_args.aipp_config,
                    # "--energy_consumption", self.all_args.energy_consumption,
                    # "--npu_id", self.all_args.npu_id,
                    # "--backend", self.all_args.backend,
                    # "--perf", self.all_args.perf,
                    # "--pipeline", self.all_args.pipeline
                                                    ]
        result = runner.invoke(benchmark_all_cmd, cmd_list)
        print(result.output)
        assert result.exit_code == 0