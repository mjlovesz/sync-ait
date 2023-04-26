# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
import logging


def get_logger(name="auto-optimizer-logger"):
    auto_logger = logging.getLogger(name)
    auto_logger.propagate = False
    auto_logger.setLevel(logging.INFO)
    if not auto_logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        auto_logger.addHandler(stream_handler)
    return auto_logger


LOG_LEVEL = {
    "notest": logging.NOTSET,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARN,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.FATAL,
    "critical": logging.CRITICAL
}


def set_logger_level(level="info"):
    if level.lower() in LOG_LEVEL:
        logger.setLevel(LOG_LEVEL.get(level.lower()))
    else:
        logger.warning("Set %s log level failed.", level)


logger = get_logger()
