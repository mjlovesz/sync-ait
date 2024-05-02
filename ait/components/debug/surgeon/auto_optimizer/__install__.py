from components.utils.install import AitInstaller
import pkg_resources

class SurgeonInstall(AitInstaller):
    def check(self):
        check_res = []
        installed_pkg = [pkg.key for pkg in pkg_resources.working_set]

        if "ait-benchmark" not in installed_pkg:
            check_res.append("[warnning] ait-benchmark not installed. will make the inference feature unusable. "
                             "use `ait install benchmark` to try again")

        if not check_res:
            return "OK"
        else:
            return "\n".join(check_res)
