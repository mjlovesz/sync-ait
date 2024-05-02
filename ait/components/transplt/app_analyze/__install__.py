import os
import sys
from components.utils.install import AitInstaller
import subprocess

class TranspltInstall(AitInstaller):
    def check(self):
        check_res = []

        if not os.path.exists(os.path.join(os.path.dirname(__file__), "headers")):
            check_res.append("[error] download data failed. will make the transplt feature unusable. "
                             "use `ait --build-extra ait-transplt` to try again")
        
        if not check_res:
            return "OK"
        else:
            return "\n".join(check_res)

    def build_extra(self, find_links=None):
        if find_links is not None:
            os.environ['AIT_INSTALL_FIND_LINKS'] = os.path.realpath(find_links)
        if sys.platform == 'win32':
            subprocess.run(["cmd", os.path.join(os.path.dirname(__file__), "install.bat")]) 
        else:
            subprocess.run(["bash", os.path.join(os.path.dirname(__file__), "install.sh")])

    def download_extra(self, dest):
        os.environ['AIT_DOWNLOAD_PATH'] =  os.path.realpath(dest)
        subprocess.run(["bash", os.path.join(os.path.dirname(__file__), "install.sh")])
        if sys.platform == 'win32':
            subprocess.run(["cmd", os.path.join(os.path.dirname(__file__), "install.bat")]) 
        else:
            subprocess.run(["bash", os.path.join(os.path.dirname(__file__), "install.sh")])

