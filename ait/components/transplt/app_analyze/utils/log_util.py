# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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
import logging
from logging import handlers

IS_PYTHON3 = sys.version_info > (3,)
LOG_FILE_PATH = "ait_transplt.log"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[%(lineno)d] - %(message)s"
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


def get_logger():
    inner_logger = logging.getLogger("ait transplt")
    inner_logger.propagate = False
    inner_logger.setLevel(logging.INFO)
    if not inner_logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT)
        stream_handler.setFormatter(formatter)
        inner_logger.addHandler(stream_handler)
    return inner_logger


def set_logger_level(level="info"):
    if level.lower() in LOG_LEVEL:
        logger.setLevel(LOG_LEVEL.get(level.lower()))
    else:
        logger.warning("Set %s log level failed.", level)


def init_file_logger():
    for ii in logger.handlers:
        # Check if already set
        if isinstance(ii, handlers.TimedRotatingFileHandler) and os.path.basename(ii.stream.name) == LOG_FILE_PATH:
            return

    if os.path.exists(LOG_FILE_PATH):
        os.remove(LOG_FILE_PATH)

    # create console handler and formatter for logger
    fh = handlers.TimedRotatingFileHandler(LOG_FILE_PATH, when='midnight', interval=1, backupCount=7)
    formatter = logging.Formatter(LOG_FORMAT)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


logger = get_logger()
