from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import logging.handlers

IS_PYTHON3 = sys.version_info > (3,)
logger = logging.getLogger('porting_advisor')

if os.path.exists("porting_advisor.log"):
    os.remove("porting_advisor.log")

# create console handler and formatter for logger
console = logging.StreamHandler()
fh = logging.handlers.TimedRotatingFileHandler("porting_advisor.log", when='midnight', interval=1, backupCount=7)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[%(lineno)d] - %(message)s")

console.setFormatter(formatter)
fh.setFormatter(formatter)

logger.addHandler(console)
logger.addHandler(fh)
