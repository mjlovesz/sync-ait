import os
import sys
import pkg_resources
from components.utils.install import AitInstaller
import subprocess

class LlmInstall(AitInstaller):
    def check(self):
        check_res = []

        if not os.path.exists(os.path.join(os.path.dirname(__file__), "opcheck/libopchecker.so")):
            check_res.append("[warnning] build libopchecker.so failed. will make the opchecker feature unusable. "
                             "use `ait build-extra llm` to try again")
        
        if not check_res:
            return "OK"
        else:
            return "\n".join(check_res)

    def build_extra(self, find_links=None):
        if sys.platform == 'win32':
            return

        subprocess.run(["bash", os.path.join(os.path.dirname(__file__), "install.sh")])
