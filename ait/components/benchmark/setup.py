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
from setuptools import setup, find_packages  # type: ignore


with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

ait_sub_tasks = [{
    "name": "benchmark",
    "help_info": "benchmark tool to get performance data including latency and throughput",
    "module": "ais_bench.infer.main_cli",
    "attr": "get_cmd_instance"
}]

ait_sub_task_entry_points = [
    f"{t.get('name')}:{t.get('help_info')} = {t.get('module')}:{t.get('attr')}"
    for t in ait_sub_tasks
]

setup(
    name='ais_bench',
    version='0.0.2',
    description='ais_bench tool',
    long_description=long_description,
    url='ais_bench url',
    packages=find_packages(),
    package_data={'': ['LICENSE', 'README.md', 'requirements.txt', 'install.bat', 'install.sh', '*.cpp', '*.h']},
    include_package_data=True,
    keywords='ais_bench tool',
    install_requires=required,
    python_requires='>=3.7',
    entry_points={
        'ait_sub_task': ait_sub_task_entry_points,
        'ait_sub_task_installer': ['ait-benchmark=ais_bench.__install__:BenchmarkInstall'],
    },
)