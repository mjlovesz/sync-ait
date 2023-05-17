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
from setuptools import setup, find_packages  # type: ignore


with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ais_bench',
    version='0.0.2',
    description='ais_bench tool',
    long_description=long_description,
    url='ais_bench url',
    packages=find_packages(),
    keywords='ais_bench tool',
    install_requires=required,
    python_requires='>=3.7',
    entry_points={
        'profile_sub_task': ['benchmark=ais_bench.infer.main_cli:benchmark_cli']
    }
)