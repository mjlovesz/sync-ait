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

import logging
import time
from configparser import ConfigParser
import numpy as np
import aclruntime

SRC_IMAGE_SIZE_W_MIN = 2
SRC_IMAGE_SIZE_W_MAX = 4096
SRC_IMAGE_SIZE_H_MIN = 1
SRC_IMAGE_SIZE_H_MAX = 4096
RBUV_SWAP_SWITCH_OFF = 0
RBUV_SWAP_SWITCH_ON = 1
AX_SWAP_SWITCH_OFF = 0
AX_SWAP_SWITCH_ON = 1
CSC_SWITCH_OFF = 0
CSC_SWITCH_ON = 0
CSC_MATRIX_MIN = -32677
CSC_MATRIX_MAX = 32676
CROP_SWITCH_OFF = 0
CROP_SWITCH_ON = 1
LOAD_START_POS_W_MIN = 0
LOAD_START_POS_W_MAX = 4095
LOAD_START_POS_H_MIN = 0
LOAD_START_POS_H_MAX = 4095
CROP_POS_W_MIN = 1
CROP_POS_W_MAX = 4096
CROP_POS_H_MIN = 1
CROP_POS_H_MAX = 4096
PADDING_SWITCH_OFF = 0
PADDING_SWITCH_ON = 1
PADDING_SIZE_MIN = 0
PADDING_SIZE_MAX = 32
PIXEL_MEAN_CHN_MIN = 0
PIXEL_MEAN_CHN_MAX = 255
PIXEL_MIN_CHN_MIN = 0
PIXEL_MIN_CHN_MAX = 255
PIXEL_VAR_RECI_CHN_MIN = -65504
PIXEL_VAR_RECI_CHN_MAX = 65504

logger = logging.getLogger(__name__)


class InferSession:
    def __init__(self, device_id: int, model_path: str, acl_json_path: str = None, debug: bool = False, loop: int = 1):
        """
        init InferSession

        Args:
            device_id: device id for npu device
            model_path: om model path to load
            acl_json_path: set acl_json_path to enable profiling or dump function
            debug: enable debug log.  Default: False
            loop: loop count for one inference. Default: 1
        """
        self.device_id = device_id
        self.model_path = model_path
        self.loop = loop
        options = aclruntime.session_options()
        if acl_json_path is not None:
            options.acl_json_path = acl_json_path
        options.log_level = 1 if debug else 2
        options.loop = self.loop
        self.session = aclruntime.InferenceSession(self.model_path, self.device_id, options)
        self.outputs_names = [meta.name for meta in self.session.get_outputs()]
        self.intensors_desc = self.session.get_inputs()
        self.outtensors_desc = self.session.get_outputs()

    @staticmethod
    def convert_tensors_to_host(tensors):
        for tensor in tensors:
            tensor.to_host()

    @staticmethod
    def convert_tensors_to_arrays(tensors):
        arrays = []
        for tensor in tensors:
            # convert acltensor to numpy array
            arrays.append(np.array(tensor))
        return arrays

    def get_inputs(self):
        """
        get inputs info of model
        """
        self.intensors_desc = self.session.get_inputs()
        return self.intensors_desc

    def get_outputs(self):
        """
        get outputs info of model
        """
        self.outtensors_desc = self.session.get_outputs()
        return self.outtensors_desc

    def set_loop_count(self, loop):
        options = self.session.options()
        options.loop = loop

    # 默认设置为静态batch
    def set_staticbatch(self):
        self.session.set_staticbatch()

    def set_dynamic_batchsize(self, dym_batch: str):
        self.session.set_dynamic_batchsize(dym_batch)

    def set_dynamic_hw(self, w: int, h: int):
        self.session.set_dynamic_hw(w, h)

    def get_max_dym_batchsize(self):
        return self.session.get_max_dym_batchsize()

    def set_dynamic_dims(self, dym_dims: str):
        self.session.set_dynamic_dims(dym_dims)

    def set_dynamic_shape(self, dym_shape: str):
        self.session.set_dynamic_shape(dym_shape)

    def set_custom_outsize(self, custom_sizes):
        self.session.set_custom_outsize(custom_sizes)

    def create_tensor_from_fileslist(self, desc, files):
        return self.session.create_tensor_from_fileslist(desc, files)

    def create_tensor_from_arrays_to_device(self, arrays):
        tensor = aclruntime.Tensor(arrays)
        tensor.to_device(self.device_id)
        return tensor

    def get_dym_aipp_input_exist(self):
        return self.session.get_dym_aipp_input_exist()

    def check_dym_aipp_input_exist(self):
        self.session.check_dym_aipp_input_exist()

    def load_aipp_config_file(self, config_file, batchsize):
        cfg = ConfigParser()
        cfg.read(config_file, 'UTF-8')
        session_list = cfg.sections()
        #多个aipp输入不支持
        if (session_list.count('aipp_op') != 1):
            logger.error("nums of section aipp_op in .config file is not supported, please check it!")
            raise RuntimeError('wrong aipp config file content!')
        option_list = cfg.options('aipp_op')
        if (option_list.count('input_format') == 1):
            self.aipp_set_input_format(cfg)
        else:
            logger.error("can not find input_format in config file, please check it!")
            raise RuntimeError('wrong aipp config file content!')

        if (option_list.count('src_image_size_w') == 1 and option_list.count('src_image_size_h') == 1):
            self.aipp_set_src_image_size(cfg)
        else:
            logger.error("can not find src_image_size in config file, please check it!")
            raise RuntimeError('wrong aipp config file content!')
        self.session.aipp_set_max_batch_size(batchsize)
        self.aipp_set_rbuv_swap_switch(cfg, option_list)
        self.aipp_set_ax_swap_switch(cfg, option_list)
        self.aipp_set_csc_params(cfg, option_list)
        self.aipp_set_crop_params(cfg, option_list)
        self.aipp_set_padding_params(cfg, option_list)
        self.aipp_set_dtc_pixel_mean(cfg, option_list)
        self.aipp_set_dtc_pixel_min(cfg, option_list)
        self.aipp_set_pixel_var_reci(cfg, option_list)

        ret = self.session.set_dym_aipp_info_set()
        return ret

    def aipp_set_input_format(self, cfg):
        input_format = cfg.get('aipp_op', 'input_format')
        legal_format = ["YUV420SP_U8", "XRGB8888_U8", "RGB888_U8", "YUV400_U8"]
        if (legal_format.count(input_format) == 1):
            self.session.aipp_set_input_format(input_format)
        else:
            logger.error("input_format in config file is illegal, please check it!")
            raise RuntimeError('wrong aipp config file content!')

    def aipp_set_src_image_size(self, cfg):
        src_image_size = list()
        tmp_size_w = cfg.getint('aipp_op', 'src_image_size_w')
        tmp_size_h = cfg.getint('aipp_op', 'src_image_size_h')
        if (SRC_IMAGE_SIZE_W_MIN <= tmp_size_w <= SRC_IMAGE_SIZE_W_MAX):
            src_image_size.append(tmp_size_w)
        else:
            logger.error("src_image_size_w in config file out of range, please check it!")
            raise RuntimeError('wrong aipp config file content!')
        if (SRC_IMAGE_SIZE_H_MIN <= tmp_size_h <= SRC_IMAGE_SIZE_H_MAX):
            src_image_size.append(tmp_size_h)
        else:
            logger.error("src_image_size_h in config file out of range, please check it!")
            raise RuntimeError('wrong aipp config file content!')

        self.session.aipp_set_src_image_size(src_image_size)

    def aipp_set_rbuv_swap_switch(self, cfg, option_list):
        if (option_list.count('rbuv_swap_switch') == 0):
            self.session.aipp_set_rbuv_swap_switch(RBUV_SWAP_SWITCH_OFF)
            return
        tmp_rs_switch = cfg.getint('aipp_op', 'rbuv_swap_switch')
        if (tmp_rs_switch == RBUV_SWAP_SWITCH_OFF or tmp_rs_switch == RBUV_SWAP_SWITCH_ON):
            self.session.aipp_set_rbuv_swap_switch(tmp_rs_switch)
        else:
            logger.error("rbuv_swap_switch in config file out of range, please check it!")
            raise RuntimeError('wrong aipp config file content!')

    def aipp_set_ax_swap_switch(self, cfg, option_list):
        if (option_list.count('ax_swap_switch') == 0):
            self.session.aipp_set_ax_swap_switch(AX_SWAP_SWITCH_OFF)
            return
        tmp_as_switch = cfg.getint('aipp_op', 'ax_swap_switch')
        if (tmp_as_switch == AX_SWAP_SWITCH_OFF or tmp_as_switch == AX_SWAP_SWITCH_ON):
            self.session.aipp_set_ax_swap_switch(tmp_as_switch)
        else:
            logger.error("ax_swap_switch in config file out of range, please check it!")
            raise RuntimeError('wrong aipp config file content!')

    def aipp_set_csc_params(self, cfg, option_list):
        if (option_list.count('csc_switch') == 0):
            tmp_csc_switch = CSC_SWITCH_OFF
        else:
            tmp_csc_switch = cfg.getint('aipp_op', 'csc_switch')

        if (tmp_csc_switch == CSC_SWITCH_OFF):
            tmp_csc_params = [0] * 16
        elif (tmp_csc_switch == CSC_SWITCH_ON):
            tmp_csc_params = list()
            tmp_csc_params.append(tmp_csc_switch)
            tmp_csc_params.append(
                0 if option_list.count('matrix_r0c0') == 0 else cfg.getint('aipp_op', 'matrix_r0c0')
            )
            tmp_csc_params.append(
                0 if option_list.count('matrix_r0c1') == 0 else cfg.getint('aipp_op', 'matrix_r0c1')
            )
            tmp_csc_params.append(
                0 if option_list.count('matrix_r0c2') == 0 else cfg.getint('aipp_op', 'matrix_r0c2')
            )
            tmp_csc_params.append(
                0 if option_list.count('matrix_r1c0') == 0 else cfg.getint('aipp_op', 'matrix_r1c0')
            )
            tmp_csc_params.append(
                0 if option_list.count('matrix_r1c1') == 0 else cfg.getint('aipp_op', 'matrix_r1c1')
            )
            tmp_csc_params.append(
                0 if option_list.count('matrix_r1c2') == 0 else cfg.getint('aipp_op', 'matrix_r1c2')
            )
            tmp_csc_params.append(
                0 if option_list.count('matrix_r2c0') == 0 else cfg.getint('aipp_op', 'matrix_r2c0')
            )
            tmp_csc_params.append(
                0 if option_list.count('matrix_r2c1') == 0 else cfg.getint('aipp_op', 'matrix_r2c1')
            )
            tmp_csc_params.append(
                0 if option_list.count('matrix_r2c2') == 0 else cfg.getint('aipp_op', 'matrix_r2c2')
            )
            tmp_csc_params.append(
                0 if option_list.count('output_bias_0') == 0 else cfg.getint('aipp_op', 'output_bias_0')
            )
            tmp_csc_params.append(
                0 if option_list.count('output_bias_1') == 0 else cfg.getint('aipp_op', 'output_bias_1')
            )
            tmp_csc_params.append(
                0 if option_list.count('output_bias_2') == 0 else cfg.getint('aipp_op', 'output_bias_2')
            )
            tmp_csc_params.append(
                0 if option_list.count('input_bias_0') == 0 else cfg.getint('aipp_op', 'input_bias_0')
            )
            tmp_csc_params.append(
                0 if option_list.count('input_bias_1') == 0 else cfg.getint('aipp_op', 'input_bias_1')
            )
            tmp_csc_params.append(
                0 if option_list.count('input_bias_2') == 0 else cfg.getint('aipp_op', 'input_bias_2')
            )

            range_ok = True
            for i in range (1, 9):
                range_ok = range_ok and (CSC_MATRIX_MIN <= tmp_csc_params[i] <= CSC_MATRIX_MAX)
            for i in range (10, 15):
                range_ok = range_ok and (0 <= tmp_csc_params[i] <= 255)
            if (range_ok is False):
                logger.error("csc_params in config file out of range, please check it!")
                raise RuntimeError('wrong aipp config file content!')
        else:
            logger.error("csc_switch in config file out of range, please check it!")
            raise RuntimeError('wrong aipp config file content!')

        self.session.aipp_set_csc_params(tmp_csc_params)

    def aipp_set_crop_params(self, cfg, option_list):
        if (option_list.count('crop') == 0):
            tmp_crop_switch = CROP_SWITCH_OFF
        else:
            tmp_crop_switch = cfg.getint('aipp_op', 'crop')

        if (tmp_crop_switch == CROP_SWITCH_OFF):
            tmp_crop_params = [0, 0, 0, 416, 416]
        elif (tmp_crop_switch == CROP_SWITCH_ON):
            tmp_crop_params = list()
            tmp_crop_params.append(tmp_crop_switch)
            tmp_crop_params.append(
                0 if option_list.count('load_start_pos_w') == 0 else cfg.getint('aipp_op', 'load_start_pos_w')
            )
            tmp_crop_params.append(
                0 if option_list.count('load_start_pos_h') == 0 else cfg.getint('aipp_op', 'load_start_pos_h')
            )
            tmp_crop_params.append(
                0 if option_list.count('crop_size_w') == 0 else cfg.getint('aipp_op', 'crop_size_w')
            )
            tmp_crop_params.append(
                0 if option_list.count('crop_size_h') == 0 else cfg.getint('aipp_op', 'crop_size_h')
            )

            range_ok = True
            range_ok = range_ok and (LOAD_START_POS_W_MIN <= tmp_crop_params[1] <= LOAD_START_POS_W_MAX)
            range_ok = range_ok and (LOAD_START_POS_H_MIN <= tmp_crop_params[2] <= LOAD_START_POS_H_MAX)
            range_ok = range_ok and (CROP_POS_W_MIN <= tmp_crop_params[3] <= CROP_POS_W_MAX)
            range_ok = range_ok and (CROP_POS_H_MIN <= tmp_crop_params[4] <= CROP_POS_H_MAX)
            if (range_ok is False):
                logger.error("crop_params in config file out of range, please check it!")
                raise RuntimeError('wrong aipp config file content!')
        else:
            logger.error("crop_switch(crop) in config file out of range, please check it!")
            raise RuntimeError('wrong aipp config file content!')

        self.session.aipp_set_crop_params(tmp_crop_params)

    def aipp_set_padding_params(self, cfg, option_list):
        if (option_list.count('padding') == 0):
            tmp_padding_switch = PADDING_SWITCH_OFF
        else:
            tmp_padding_switch = cfg.getint('aipp_op', 'padding')

        if (tmp_padding_switch == PADDING_SWITCH_OFF):
            tmp_padding_params = [0] * 5
        elif (tmp_padding_switch == PADDING_SWITCH_ON):
            tmp_padding_params = list()
            tmp_padding_params.append(tmp_padding_switch)
            tmp_padding_params.append(
                0 if option_list.count('padding_size_top') == 0 else cfg.getint('aipp_op', 'padding_size_top')
            )
            tmp_padding_params.append(
                0 if option_list.count('padding_size_bottom') == 0 else cfg.getint('aipp_op', 'padding_size_bottom')
            )
            tmp_padding_params.append(
                0 if option_list.count('padding_size_left') == 0 else cfg.getint('aipp_op', 'padding_size_left')
            )
            tmp_padding_params.append(
                0 if option_list.count('padding_size_right') == 0 else cfg.getint('aipp_op', 'padding_size_right')
            )

            range_ok = True
            for i in range (1, 5):
                range_ok = range_ok and (PADDING_SIZE_MIN <= tmp_padding_params[i] <= PADDING_SIZE_MAX)
            if (range_ok is False):
                logger.error("padding_params in config file out of range, please check it!")
                raise RuntimeError('wrong aipp config file content!')
        else:
            logger.error("padding_switch in config file out of range, please check it!")
            raise RuntimeError('wrong aipp config file content!')

        self.session.aipp_set_padding_params(tmp_padding_params)

    def aipp_set_dtc_pixel_mean(self, cfg, option_list):
        tmp_mean_params = list()
        tmp_mean_params.append(
            0 if option_list.count('mean_chn_0') == 0 else cfg.getint('aipp_op', 'mean_chn_0')
        )
        tmp_mean_params.append(
            0 if option_list.count('mean_chn_1') == 0 else cfg.getint('aipp_op', 'mean_chn_1')
        )
        tmp_mean_params.append(
            0 if option_list.count('mean_chn_2') == 0 else cfg.getint('aipp_op', 'mean_chn_2')
        )
        tmp_mean_params.append(
            0 if option_list.count('mean_chn_3') == 0 else cfg.getint('aipp_op', 'mean_chn_3')
        )

        range_ok = True
        for i in range (0, 4):
            range_ok = range_ok and (PIXEL_MEAN_CHN_MIN <= tmp_mean_params[i] <= PIXEL_MEAN_CHN_MAX)
        if (range_ok is False):
            logger.error("mean_chn_params in config file out of range, please check it!")
            raise RuntimeError('wrong aipp config file content!')

        self.session.aipp_set_dtc_pixel_mean(tmp_mean_params)

    def aipp_set_dtc_pixel_min(self, cfg, option_list):
        tmp_min_params = list()
        tmp_min_params.append(
            0 if option_list.count('min_chn_0') == 0 else cfg.getfloat('aipp_op', 'min_chn_0')
        )
        tmp_min_params.append(
            0 if option_list.count('min_chn_1') == 0 else cfg.getfloat('aipp_op', 'min_chn_1')
        )
        tmp_min_params.append(
            0 if option_list.count('min_chn_2') == 0 else cfg.getfloat('aipp_op', 'min_chn_2')
        )
        tmp_min_params.append(
            0 if option_list.count('min_chn_3') == 0 else cfg.getfloat('aipp_op', 'min_chn_3')
        )

        range_ok = True
        for i in range (0, 4):
            range_ok = range_ok and (PIXEL_MIN_CHN_MIN <= tmp_min_params[i] <= PIXEL_MIN_CHN_MAX)
        if (range_ok is False):
            logger.error("min_chn_params in config file out of range, please check it!")
            raise RuntimeError('wrong aipp config file content!')

        self.session.aipp_set_dtc_pixel_min(tmp_min_params)

    def aipp_set_pixel_var_reci(self, cfg, option_list):
        tmp_reci_params = list()
        tmp_reci_params.append(
            0 if option_list.count('var_reci_chn_0') == 0 else cfg.getfloat('aipp_op', 'var_reci_chn_0')
        )
        tmp_reci_params.append(
            0 if option_list.count('var_reci_chn_1') == 0 else cfg.getfloat('aipp_op', 'var_reci_chn_1')
        )
        tmp_reci_params.append(
            0 if option_list.count('var_reci_chn_2') == 0 else cfg.getfloat('aipp_op', 'var_reci_chn_2')
        )
        tmp_reci_params.append(
            0 if option_list.count('var_reci_chn_3') == 0 else cfg.getfloat('aipp_op', 'var_reci_chn_3')
        )

        range_ok = True
        for i in range (0, 4):
            range_ok = range_ok and (PIXEL_VAR_RECI_CHN_MIN <= tmp_reci_params[i] <= PIXEL_VAR_RECI_CHN_MAX)
        if (range_ok is False):
            logger.error("var_reci_chn_params in config file out of range, please check it!")
            raise RuntimeError('wrong aipp config file content!')

        self.session.aipp_set_dtc_pixel_min(tmp_reci_params)

    def run(self, feeds, out_array=False):
        if len(feeds) > 0 and isinstance(feeds[0], np.ndarray):
            # if feeds is ndarray list, convert to baseTensor
            inputs = []
            for array in feeds:
                basetensor = aclruntime.BaseTensor(array.__array_interface__['data'][0], array.nbytes)
                inputs.append(basetensor)
        else:
            inputs = feeds
        outputs = self.session.run(self.outputs_names, inputs)
        if out_array:
            # convert to host tensor
            self.convert_tensors_to_host(outputs)
            # convert tensor to narray
            return self.convert_tensors_to_arrays(outputs)
        else:
            return outputs

    def run_pipeline(self, infilelist, output, auto_shape=0, auto_dims=0):
        # print("pipeline running.....")
        self.session.run_pipeline(infilelist, output, auto_shape, auto_dims)

    def reset_sumaryinfo(self):
        self.session.reset_sumaryinfo()

    def sumary(self):
        return self.session.sumary()

    def finalize(self):
        if hasattr(self.session, 'finalize'):
            self.session.finalize()

    def infer(self, feeds, mode = 'static', custom_sizes = 100000):
        '''
        Parameters:
            feeds: input data
            mode: static dymdims dymshapes
        '''
        inputs = []
        shapes = []
        torch_tensor_list = ['torch.FloatTensor', 'torch.DoubleTensor', 'torch.HalfTensor',
            'torch.BFloat16Tensor', 'torch.ByteTensor', 'torch.CharTensor', 'torch.ShortTensor',
            'torch.LongTensor', 'torch.BoolTensor', 'torch.IntTensor' ]
        np_type_list = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.float16, \
                      np.float32, np.float64]
        for feed in feeds:
            if type(feed) is np.ndarray:
                infer_input = feed
                shapes.append(infer_input.shape)
            elif type(feed) in np_type_list:
                infer_input = np.array(feed)
                shapes.append([feed.size])
            elif type(feed) is aclruntime.Tensor:
                infer_input = feed
                shapes.append(infer_input.shape)
            elif hasattr(feed, 'type') and feed.type() in torch_tensor_list:
                infer_input = feed.numpy()
                if not feed.is_contiguous():
                    infer_input = np.ascontiguousarray(infer_input)
                shapes.append(infer_input.shape)
            else:
                raise RuntimeError('type:{} invalid'.format(type(feed)))
            inputs.append(infer_input)

        if mode == 'dymshape' or mode == 'dymdims':
            dym_list = []
            indesc = self.get_inputs()
            outdesc = self.get_outputs()
            for i, shape in enumerate(shapes):
                str_shape = [ str(val) for val in shape ]
                dyshape = "{}:{}".format(indesc[i].name, ",".join(str_shape))
                dym_list.append(dyshape)
            dyshapes = ';'.join(dym_list)
            if mode == 'dymshape':
                self.session.set_dynamic_shape(dyshapes)
                if isinstance(custom_sizes, int):
                    custom_sizes = [custom_sizes]*len(outdesc)
                elif isinstance(custom_sizes, list) is False:
                    raise RuntimeError('custom_sizes:{} type:{} invalid'.format(
                        custom_sizes, type(custom_sizes)))
                self.session.set_custom_outsize(custom_sizes)
            elif mode == 'dymdims':
                self.session.set_dynamic_dims(dyshapes)
        return self.run(inputs, out_array=True)


class MemorySummary:
    @staticmethod
    def get_h2d_time_list():
        if hasattr(aclruntime, 'MemorySummary'):
            return aclruntime.MemorySummary().H2D_time_list
        else:
            return []

    @staticmethod
    def get_d2h_time_list():
        if hasattr(aclruntime, 'MemorySummary'):
            return aclruntime.MemorySummary().D2H_time_list
        else:
            return []

    @staticmethod
    def reset():
        if hasattr(aclruntime, 'MemorySummary'):
            aclruntime.MemorySummary().reset()
