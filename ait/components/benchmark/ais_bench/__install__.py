import os
import sys
import pkg_resources
from components.utils.install import AitInstaller
import subprocess

class BenchmarkInstall(AitInstaller):
    def check(self):
        check_res = []
        installed_pkg = [pkg.key for pkg in pkg_resources.working_set]

        if "aclruntime" not in installed_pkg:
            check_res.append("[error] aclruntime not installed. use `ait --build-extra ait-benchmark` to try again")

        if not check_res:
            return "OK"
        else:
            return "\n".join(check_res)

    def build_extra(self, find_links=None):
        if sys.platform == 'win32':
            return
        
        if find_links is not None:
            os.environ['AIT_INSTALL_FIND_LINKS'] = os.path.realpath(find_links)
        subprocess.run(["bash", os.path.join(os.path.dirname(__file__), "install.sh")])

    def download_extra(self, dest):
        if sys.platform == 'win32':
            return 
        
        os.environ['AIT_DOWNLOAD_PATH'] =  os.path.realpath(dest)
        subprocess.run(["bash", os.path.join(os.path.dirname(__file__), "install.sh")])
