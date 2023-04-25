#    Copyright (c) 2023, Huawei Technologies Co., Ltd
#    All rights reserved.
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
from setuptools import setup, find_packages  # type: ignore

with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='msquikcmp',
    version='0.0.1',
    description='This tool enables one-click network-wide accuracy analysis of tensorflow and ONNX models.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://gitee.com/ascend/auto-optimizer',
    packages=find_packages(),
    package_data={'': ['LICENSE']},
    license='Apache-2.0',
    keywords='msquikcmp',
    install_requires=required,
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
    python_requires='>=3.7',
    entry_points={
        'console_scripts': ['msquikcmp=compare.__main__:main'],
        'debug_sub_task': ['compare=compare.__main__:main'],
    },
)
