class MyArgs():
    def __init__(self, model, input, output, output_dirname, outfmt, loop, debug, device, dymBatch, dymHW, dymDims, outputSize, auto_set_dymshape_mode, auto_set_dymdims_mode, batchsize, pure_data_type, profiler, dump, acl_json_path, output_batchsize_axis, run_mode, display_all_summary, warmup_count, dymShape_range):
        self.model = model
        self.input = input
        self.output = output
        self.output_dirname = output_dirname
        self.outfmt = outfmt
        self.loop = loop
        self.debug = debug
        self.device = device
        self.dymBatch = dymBatch
        self.dymHW = dymHW
        self.dymDims = dymDims
        self.outputSize = outputSize
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
        self.dymShape_range = dymShape_range
