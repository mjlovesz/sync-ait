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
import sys
import os
import subprocess
from components.utils.install import AitInstaller


class ConvertInstall(AitInstaller):
    def check(self):
        check_res = []

        if not os.path.exists(os.path.join(os.path.dirname(__file__), "aie_convert")):
            check_res.append("[warnning] build aie_convert failed. will make the AIE feature unusable. "
                             "use `ait build-extra convert` to try again")
        
        if not check_res:
            return "OK"
        else:
            return "\n".join(check_res)
        
    def build_extra(self, find_links=None):
        if sys.platform == 'win32':
            return

        subprocess.run(["bash", os.path.abspath(os.path.join(os.path.dirname(__file__), "install.sh"))], shell=False)