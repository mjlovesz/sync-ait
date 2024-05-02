from components.utils.install import AitInstaller
import sys
import os
import subprocess

class ConvertInstall(AitInstaller):
    def check(self):
        check_res = []

        if not os.path.exists(os.path.join(os.path.dirname(__file__), "aie_convert")):
            check_res.append("[warnning] build aie_convert failed. will make the AIE feature unusable. "
                             "use `ait --build-extra ait-convert` to try again")
        
        if not check_res:
            return "OK"
        else:
            return "\n".join(check_res)
        
    def build_extra(self, find_links=None):
        if sys.platform == 'win32':
            return

        subprocess.run(["bash", os.path.join(os.path.dirname(__file__), "install.sh")])
