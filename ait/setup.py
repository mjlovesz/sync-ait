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

from setuptools import setup, find_packages


abs_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(abs_path, "requirements.txt")) as f:
    required = f.read().splitlines()

setup(
    name='ait',
    version='0.0.1',
    description='AIT, Ascend Inference Tools',
    long_description_content_type='text/markdown',
    url='https://gitee.com/ascend/ait',
    packages=find_packages(),
    package_data={'': ['LICENSE']},
    license='Apache-2.0',
    keywords='ait',
    python_requires='>=3.7',
    install_requires=required,
    entry_points={
        'console_scripts': ['ait=components.__main__:main'],
    },
)
