# Copyright 2023 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from setuptools import setup, find_packages


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
    extras_require={
        'profile': [
            (
                'aclruntime @ git+https://gitee.com/ascend/ait.git'
                '#egg=aclruntime&subdirectory=ait/components/profile/benchmark/backend'
            ),
            (
                'ais_bench @ git+https://gitee.com/ascend/ait.git'
                '#egg=ais_bench&subdirectory=ait/components/profile/benchmark'
            ),
        ],
        'debug': [
            (
                'compare @ git+https://gitee.com/ascend/ait.git'
                '#egg=compare&subdirectory=ait/components/debug/compare'
            ),            
            (
                'auto_optimizer @ git+https://gitee.com/ascend/ait.git'
                '#egg=auto_optimizer&subdirectory=ait/components/debug/surgeon'
            ),
        ],
        'analyze': [
            (
                'app_analysis @ git+https://gitee.com/ascend/ait.git'
                '#egg=app_analysis&subdirectory=ait/components/analyze/transplt'
            )
        ]
    },
    entry_points={
        'console_scripts': ['ait=components.__main__:cli'],
    },
)
