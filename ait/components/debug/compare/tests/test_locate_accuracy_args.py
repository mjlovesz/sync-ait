
import logging
import sys
import os
import pytest
from msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter
from msquickcmp.accuracy_locat.locat_accuracy import find_accuracy_interval

logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = '[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


