import os
import sys
import pkg_resources
from components.utils.install import AitInstaller
import subprocess

class CompareInstall(AitInstaller):
    def check(self):
        check_res = []
        installed_pkg = [pkg.key for pkg in pkg_resources.working_set]

        if "ait-benchmark" not in installed_pkg:
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
        
        subprocess.run(["bash", os.path.join(os.path.dirname(__file__), "install.sh")])
