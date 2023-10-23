import os
import sys
import logging
import numpy as np
import pytest
import aclruntime
from ais_bench.infer.interface import InferSession, MultiDeviceSession
from test_common import TestCommonClass


logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = '[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class TestClass:
    @classmethod
    def setup_class(cls):
        """
        class level setup_class
        """
        cls.init(TestClass)

    @classmethod
    def teardown_class(cls):
        logger.info('\n ---class level teardown_class')

    @classmethod
    def get_resnet50_static(cls, bs:int = 1):
        return os.path.join(
            TestCommonClass.base_path, cls.resnet_name, "model", f"pth_resnet50_bs{bs}.om"
        )

    @classmethod
    def get_resnet50_dynamic(cls, kind:str):
        return os.path.join(
            TestCommonClass.base_path, cls.resnet_name, "model", f"pth_resnet50_{kind}.om"
        )

    @classmethod
    def get_add_model_static(cls, bs:int = 1):
        return os.path.join(
            TestCommonClass.base_path, cls.add_name, "model", f"add_model_bs{bs}.om"
        )

    @classmethod
    def get_add_model_dynamic(cls, kind:str):
        return os.path.join(
            TestCommonClass.base_path, cls.add_name, "model", f"add_model_{kind}.om"
        )

    def init(self):
        self.resnet_name = "resnet50"
        self.add_name = "add_model"
        self.iteration_times = 1000
        self.in_out_list = [-1, 0]

# ====================test single process infer api==========================
    def test_infer_api_static(self):
        device_id = 0
        model_path = self.get_resnet50_static(1)
        session = InferSession(device_id, model_path)

        # create new numpy data according inputs info
        shape = session.get_inputs()[0].shape
        ndata = np.full(shape, 0).astype(np.uint8)

        # in is numpy list and output is numpy list
        outputs = session.infer([ndata], mode='static')
        session.free_resource()

    def test_infer_api_dymbatch(self):
        device_id = 0
        model_path = self.get_resnet50_dynamic('dymbatch')
        session = InferSession(device_id, model_path)

        # create new numpy data according inputs info
        shape = [1,3,256,256]
        ndata = np.full(shape, 0).astype(np.uint8)

        # in is numpy list and output is numpy list
        outputs = session.infer([ndata], mode='dymbatch')
        session.free_resource()

    def test_infer_api_dymwh(self):
        device_id = 0
        model_path = self.get_resnet50_dynamic('dymwh')
        session = InferSession(device_id, model_path)

        # create new numpy data according inputs info
        shape = [1,3,224,224]
        ndata = np.full(shape, 0).astype(np.float32)

        # in is numpy list and output is numpy list
        outputs = session.infer([ndata], mode='dymhw')
        session.free_resource()

    def test_infer_api_dymdim(self):
        device_id = 0
        model_path = self.get_resnet50_dynamic('dymdim')
        session = InferSession(device_id, model_path)

        # create new numpy data according inputs info
        shape = [1,3,224,224]
        ndata = np.full(shape, 0).astype(np.float32)

        # in is numpy list and output is numpy list
        outputs = session.infer([ndata], mode='dymdims')
        session.free_resource()

    def test_infer_api_dymshape(self):
        device_id = 0
        model_path = self.get_resnet50_dynamic('dymshape')
        session = InferSession(device_id, model_path)

        # create new numpy data according inputs info
        shape = [1,3,224,224]
        ndata = np.full(shape, 0).astype(np.float32)

        # in is numpy list and output is numpy list
        outputs = session.infer([ndata], mode='dymshape', custom_sizes=100000)
        session.free_resource()

# ====================test single process infer pipeline api==========================
    def test_infer_pipeline_api_static(self):
        device_id = 0
        model_path = self.get_resnet50_static(1)
        session = InferSession(device_id, model_path)

        # create new numpy data according inputs info
        shape = session.get_inputs()[0].shape
        ndata = np.full(shape, 0).astype(np.uint8)
        ndata_list = [[ndata], [ndata], [ndata]]

        # in is numpy list and output is numpy list
        outputs = session.infer_pipeline(ndata_list, mode='static')
        session.free_resource()

    def test_infer_pipeline_api_dymbatch(self):
        device_id = 0
        model_path = self.get_resnet50_dynamic('dymbatch')
        session = InferSession(device_id, model_path)

        # create new numpy data according inputs info
        shape = [1,3,256,256]
        ndata = np.full(shape, 0).astype(np.uint8)
        ndata_list = [[ndata], [ndata], [ndata]]

        # in is numpy list and output is numpy list
        outputs = session.infer_pipeline(ndata_list, mode='dymbatch')
        session.free_resource()

    def test_infer_pipeline_api_dymwh(self):
        device_id = 0
        model_path = self.get_resnet50_dynamic('dymwh')
        session = InferSession(device_id, model_path)

        # create new numpy data according inputs info
        shape = [1,3,224,224]
        ndata = np.full(shape, 0).astype(np.float32)
        ndata_list = [[ndata], [ndata], [ndata]]

        # in is numpy list and output is numpy list
        outputs = session.infer_pipeline(ndata_list, mode='dymhw')
        session.free_resource()

    def test_infer_pipeline_api_dymdim(self):
        device_id = 0
        model_path = self.get_resnet50_dynamic('dymdim')
        session = InferSession(device_id, model_path)

        # create new numpy data according inputs info
        shape1 = [1,3,224,224]
        shape2 = [8,3,448,448]
        ndata1 = np.full(shape1, 0).astype(np.float32)
        ndata2 = np.full(shape2, 0).astype(np.float32)
        ndata_list = [[ndata1], [ndata2]]
        # in is numpy list and output is numpy list
        outputs = session.infer_pipeline(ndata_list, mode='dymdims')
        session.free_resource()

    def test_infer_pipeline_api_dymshape(self):
        device_id = 0
        model_path = self.get_resnet50_dynamic('dymshape')
        session = InferSession(device_id, model_path)

        # create new numpy data according inputs info
        shape1 = [1,3,224,224]
        shape2 = [4,3,225,225]
        ndata1 = np.full(shape1, 0).astype(np.float32)
        ndata2 = np.full(shape2, 0).astype(np.float32)
        ndata_list = [[ndata1], [ndata2]]

        # in is numpy list and output is numpy list
        outputs = session.infer_pipeline(ndata_list, mode='dymshape', custom_sizes=100000)
        session.free_resource()

# ====================test single process infer iteration api==========================
    def test_infer_iteration_api_static(self):
        device_id = 0
        model_path = self.get_add_model_static(1)
        session = InferSession(device_id, model_path)

        # create new numpy data according inputs info
        shape = session.get_inputs()[0].shape
        ndata = np.full(shape, 0).astype(np.float32)

        # in is numpy list and output is numpy list
        outputs = session.infer_iteration([ndata, ndata], in_out_list=self.in_out_list,
            iteration_times=self.iteration_times, mode='static')
        session.free_resource()

    def test_infer_iteration_api_dymbatch(self):
        device_id = 0
        model_path = self.get_add_model_dynamic('dymbatch')
        session = InferSession(device_id, model_path)

        # create new numpy data according inputs info
        shape = [1,3,32,32]
        ndata = np.full(shape, 0).astype(np.float32)

        # in is numpy list and output is numpy list
        outputs = session.infer_iteration([ndata, ndata], in_out_list=self.in_out_list,
            iteration_times=self.iteration_times, mode='dymbatch')
        session.free_resource()

    def test_infer_iteration_api_dymwh(self):
        device_id = 0
        model_path = self.get_add_model_dynamic('dymwh')
        session = InferSession(device_id, model_path)

        # create new numpy data according inputs info
        shape = [1,3,32,32]
        ndata = np.full(shape, 0).astype(np.float32)

        # in is numpy list and output is numpy list
        outputs = session.infer_iteration([ndata, ndata], in_out_list=self.in_out_list,
            iteration_times=self.iteration_times, mode='dymhw')
        session.free_resource()

    def test_infer_iteration_api_dymdim(self):
        device_id = 0
        model_path = self.get_add_model_dynamic('dymdim')
        session = InferSession(device_id, model_path)

        # create new numpy data according inputs info
        shape = [4,3,64,64]
        ndata = np.full(shape, 0).astype(np.float32)

        # in is numpy list and output is numpy list
        outputs = session.infer_iteration([ndata, ndata], in_out_list=self.in_out_list,
            iteration_times=self.iteration_times, mode='dymdims')
        session.free_resource()

    def test_infer_iteration_api_dymshape(self):
        device_id = 0
        model_path = self.get_add_model_dynamic('dymshape')
        session = InferSession(device_id, model_path)

        # create new numpy data according inputs info
        shape = [4,3,32,32]
        ndata = np.full(shape, 0).astype(np.float32)
        out_size = 4 * 3 * 32 * 32 * 4
        # in is numpy list and output is numpy list
        outputs = session.infer_iteration([ndata, ndata], in_out_list=self.in_out_list,
            iteration_times=self.iteration_times, mode='dymshape', custom_sizes=out_size)
        session.free_resource()
