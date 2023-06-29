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
@opt_pipeline
def benchmark_all_cmd(om_model,
                     input_path,
                     output,
                     output_dirname,
                     outfmt,
                     loop,
                     debug,
                     device,
                     dym_batch,
                     dym_hw,
                     dym_dims,
                     dym_shape,
                     output_size,
                     auto_set_dymshape_mode,
                     auto_set_dymdims_mode,
                     batch_size,
                     pure_data_type,
                     profiler,
                     dump,
                     acl_json_path,
                     output_batchsize_axis,
                     run_mode,
                     display_all_summary,
                     warmup_count,
                     dym_shape_range,
                     aipp_config,
                     energy_consumption,
                     npu_id,
                     backend,
                     perf,
                     pipeline
                     ):
     click.echo('Hello %s!' % om_model)

class TestClass:
    @classmethod
    def setup_class(cls):
        """
        class level setup_class
        """
        cls.init(TestClass)

    @classmethod
    def get_click_list(cls, cmd_adapter:BenchMarkArgsAdapter):
        cmd_list = ["--om-model", cmd_adapter.model,
                    "--input", cmd_adapter.input,
                    "--output", cmd_adapter.output,
                    "--output-dirname", cmd_adapter.output_dirname,
                    "--outfmt", cmd_adapter.outfmt,
                    "--loop", cmd_adapter.loop,
                    "--debug", cmd_adapter.debug,
                    "--device", cmd_adapter.device,
                    "--dym-batch", cmd_adapter.dym_batch,
                    "--dym-hw", cmd_adapter.dym_hw,
                    "--dym-dims", cmd_adapter.dym_dims,
                    "--dym-shape", cmd_adapter.dym_shape,
                    "--output-size", cmd_adapter.output_size,
                    "--auto-set-dymshape-mode", cmd_adapter.auto_set_dymshape_mode,
                    "--auto-set-dymdims-mode", cmd_adapter.auto_set_dymdims_mode,
                    "--batch-size", cmd_adapter.batchsize,
                    "--pure-data-type", cmd_adapter.pure_data_type,
                    "--profiler", cmd_adapter.profiler,
                    "--dump", cmd_adapter.dump,
                    "--acl-json-path", cmd_adapter.acl_json_path,
                    "--output-batchsize-axis", cmd_adapter.output_batchsize_axis,
                    "--run-mode", cmd_adapter.run_mode,
                    "--display-all-summary", cmd_adapter.display_all_summary,
                    "--warmup-count", cmd_adapter.warmup_count,
                    "--dym-shape-range", cmd_adapter.dym_shape_range,
                    "--aipp-config", cmd_adapter.aipp_config,
                    "--energy_consumption", cmd_adapter.energy_consumption,
                    "--npu_id", cmd_adapter.npu_id,
                    "--backend", cmd_adapter.backend,
                    "--perf", cmd_adapter.perf,
                    "--pipeline", cmd_adapter.pipeline
                                                    ]
        return cmd_list

    def init(self):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.standard_args = BenchMarkArgsAdapter (
            model=os.path.join(self.current_dir, "../testdata/resnet50/model/pth_resnet50_bs4.om"),
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
        cmd_list = self.get_click_list(self.standard_args)
        result = runner.invoke(benchmark_all_cmd, cmd_list)
        print(result.output)
        assert result.exit_code == 0

    def test_check_all_simple_args_legality(self):
        runner = CliRunner()
        cmd_list = ["-om", self.standard_args.model,
                    "-i", self.standard_args.input,
                    "-o", self.standard_args.output,
                    "-od", self.standard_args.output_dirname,
                    "--outfmt", self.standard_args.outfmt,
                    "--loop", self.standard_args.loop,
                    "--debug", self.standard_args.debug,
                    "-d", self.standard_args.device,
                    "-db", self.standard_args.dym_batch,
                    "-dhw", self.standard_args.dym_hw,
                    "-dd", self.standard_args.dym_dims,
                    "-ds", self.standard_args.dym_shape,
                    "-outsize", self.standard_args.output_size,
                    "-asdsm", self.standard_args.auto_set_dymshape_mode,
                    "-asddm", self.standard_args.auto_set_dymdims_mode,
                    "--batch-size", self.standard_args.batchsize,
                    "-pdt", self.standard_args.pure_data_type,
                    "-pf", self.standard_args.profiler,
                    "--dump", self.standard_args.dump,
                    "-acl", self.standard_args.acl_json_path,
                    "-oba", self.standard_args.output_batchsize_axis,
                    "-rm", self.standard_args.run_mode,
                    "-das", self.standard_args.display_all_summary,
                    "-wcount", self.standard_args.warmup_count,
                    "-dr", self.standard_args.dym_shape_range,
                    "-aipp", self.standard_args.aipp_config,
                    "-ec", self.standard_args.energy_consumption,
                    "--npu_id", self.standard_args.npu_id,
                    "--backend", self.standard_args.backend,
                    "--perf", self.standard_args.perf,
                    "--pipeline", self.standard_args.pipeline
                                                    ]
        result = runner.invoke(benchmark_all_cmd, cmd_list)
        print(result.output)
        assert result.exit_code == 0
