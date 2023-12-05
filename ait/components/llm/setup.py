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

import subprocess
import site

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CustomBuildExt(build_ext):
    def run(self):
        subprocess.check_call(['bash', 'llm/dump/backend/build.sh'], shell=False)
        super().run()


setup(
    name='ait-llm',
    version='0.1.0',
    description='Debug tools for large language model(llm)',
    url='https://gitee.com/ascend/ait/ait/components/llm',
    packages=find_packages(),
    license='Apache-2.0',
    keywords='llm',
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
    cmdclass={'build_ext': CustomBuildExt},
    data_dir=f"{site.getsitepackages()[0]}",
    data_files=[('llm', ['llm/dump/backend/lib/libatb_probe.so'])],
    include_package_data=True,
    python_requires='>=3.7',
    entry_points={
        'llm_sub_task': ['llm=llm.__main__:get_cmd_instance'],
    },
)