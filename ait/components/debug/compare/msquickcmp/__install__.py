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
import os
import sys
import subprocess
import pkg_resources
from components.utils.install import AitInstaller


class CompareInstall(AitInstaller):
    def check(self):
        check_res = []
        installed_pkg = [pkg.key for pkg in pkg_resources.working_set]

        if "ais-bench" not in installed_pkg:
            check_res.append("[error] ait-benchmark not installed. use `ait install benchmark` to try again")

        if "ait-surgeon" not in installed_pkg:
            check_res.append("[error] ait-surgeon not installed. use `ait install surgeon` to try again")

        if not os.path.exists(os.path.join(os.path.dirname(__file__), "libsaveom.so")):
            check_res.append("[error] build lib saveom.so failed. use `ait build-extra compare` to try again")
        
        if not check_res:
            return "OK"
        else:
            return "\n".join(check_res)

    def build_extra(self, find_links=None):
        if sys.platform == 'win32':
            return
        
        subprocess.run(["/bin/bash", os.path.abspath(os.path.join(os.path.dirname(__file__), "install.sh"))], shell=False)
