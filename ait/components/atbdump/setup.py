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


from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import subprocess

class CustomBuildExt(build_ext):
    def run(self):
        subprocess.check_call(['bash', 'my_module/cpp_code/build.sh'])
        super().run()

setup(
    name='my_project',
    version='1.0',
    packages=['my_module'],
    ext_modules=[Extension('my_module.my_cpp_module', sources=['my_module/cpp_code/my_cpp_code.cpp'])],
    cmdclass={'build_ext': CustomBuildExt},
    install_requires=[],  # 添加你的依赖项
)


setup(
    name='atbdump',
    version='0.1.0',
    description='atb dump tool',
    url='https://gitee.com/ascend/ait/ait/components/atbdump',
    packages=find_packages(),
    license='Apache-2.0',
    keywords='atbdump',
    classifiers=[
        'Development Status :: Alpha',
        'Intended Audience :: Developers',
        'License :: Apache-2.0 Software License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development'
    ],
    ext_modules=[Extension('my_module.my_cpp_module', sources=['my_module/cpp_code/my_cpp_code.cpp'])],
    cmdclass={'build_ext': CustomBuildExt},
    python_requires='>=3.7',
    entry_points={
        'atbdump_sub_task': ['atbdump=dump.__main__:get_cmd_instance'],
    },
)