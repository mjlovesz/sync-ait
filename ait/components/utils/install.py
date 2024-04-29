import os
import logging
import sys
import subprocess
from abc import abstractmethod
from .util import get_entry_points
from typing import Union

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def is_windows():
    return sys.platform == "win32"


def warning_in_windows(title):
    if is_windows():
        logger.warning(f"{title} is not support windows")
        return True
    return False


def get_base_path():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(base)
    return base


def get_real_pkg_path(pkg_path):
    return os.path.join(get_base_path(), pkg_path)


class AitInstaller:
    @abstractmethod
    def check(self):
        return "OK"

    @abstractmethod
    def build_extra(self):
        logger.info("there are no more extra dependencies to build")


INSTALL_INFO_MAP = [
    {
        "pkg-name": "ait-llm",
        "pkg-path": "llm",
    },
    {
        "pkg-name": "ait-surgeon",
        "pkg-path": os.path.join("debug", "surgeon"),
        "support_windows": True
    },
    {
        "pkg-name": "ait-analyze",
        "pkg-path": "analyze",
    },
    {
        "pkg-name": "ait-transplt",
        "pkg-path": "transplt",
        "support_windows": True
    },
    {
        "pkg-name": "ait-convert",
        "pkg-path": "convert",
    },
    {
        "pkg-name": "ait-msprof",
        "pkg-path": os.path.join("profile", "msprof"),
    },
    {
        "pkg-name": "ait-benchmark",
        "pkg-path": "benchmark",
    },
    {
        "pkg-name": "ait-compare",
        "pkg-path": os.path.join("debug", "compare"),
        "depends": [
            "ait-benchmark",
            "ait-surgeon"
        ]
    },
]


def get_install_info_follow_depends(install_infos):
    all_names = set()
    for info in install_infos:
        all_names.add(info.get("pkg-name"))
        all_names.update(info.get("depends", []))
    if len(all_names) == len(install_infos):
        return install_infos
    else:
        return list(filter(lambda info: info["pkg-name"] in all_names, INSTALL_INFO_MAP))

def install_tools(names):
    if len(names) == 0:
        logger.info("You can specify the components you want to install, "
                    "you can select more than one, "
                    "or you can use --install all to install all components.")
        return 
    if "all" in names:
        install_infos = INSTALL_INFO_MAP
    else:
        install_infos = list(filter(lambda info: info["pkg-name"] in names, INSTALL_INFO_MAP))

        install_infos = get_install_info_follow_depends(install_infos)

    for tool_info in install_infos:
        install_tool(tool_info)


def install_tool(tool_info):
    pkg_name = tool_info.get("pkg-name")
    support_windows = tool_info.get("support_windows", False)
    if not support_windows and warning_in_windows(pkg_name):
        return
    logger.info(f"installing {pkg_name}")
    pkg_path = get_real_pkg_path(tool_info.get("pkg-path"))

    subprocess.run([sys.executable, "-m", "pip", "install", pkg_path])
    subprocess.run([sys.executable, "-m", "components", "--build-extra", pkg_name])


def get_installer(pkg_name) -> Union[AitInstaller, None]:
    entry_points = get_entry_points("ait_sub_task_installer")
    pkg_installer = None
    for entry_point in entry_points:
        if entry_point.name == pkg_name:
            pkg_installer = entry_point.load()()
            break
    if isinstance(pkg_installer, AitInstaller):
        return pkg_installer
    return None

def check_tools(names):
    if "all" in names or len(names) == 0:
        install_infos = INSTALL_INFO_MAP
    else:
        install_infos = filter(lambda info: info["pkg-name"] in names, INSTALL_INFO_MAP)

    for tool_info in install_infos:
        pkg_name = tool_info.get("pkg-name")
        logger.info(pkg_name)
        for msg in check_tool(pkg_name).split("\n"):
            logger.info(f"  {msg}")

def check_tool(pkg_name):
    logger.debug(f"checking {pkg_name}")
    pkg_installer = get_installer(pkg_name)

    if not pkg_installer:
        return "not install yet."
    else:
        return pkg_installer.check()


def build_extra(pkg_name):
    logger.info(f"installing extra of {pkg_name}")
    pkg_installer = get_installer(pkg_name)

    if not pkg_installer:
        pkg_installer = AitInstaller()
    return pkg_installer.build_extra()
