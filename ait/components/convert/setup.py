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

ait_sub_tasks = [{
    "name": "convert",
    "help_info": "convert tool converts the model from ONNX, TensorFlow, Caffe and MindSpore to OM. \
                   It supports atc, aoe and aie.",
    "module": "model_convert.__main__",
    "attr": "get_cmd_instance"
}]

ait_sub_task_entry_points = [
    f"{t.get('name')}:{t.get('help_info')} = {t.get('module')}:{t.get('attr')}"
    for t in ait_sub_tasks
]

setup(
    name='ait-convert',
    version='7.0.0c2',
    description='model convert tool',
    url='https://gitee.com/ascend/ait',
    packages=find_packages(),
    package_data={'': ['LICENSE', 'README.md', 'requirements.txt', 'install.bat', 'install.sh', '*.cpp', '*.h']},
    license='Apache-2.0',
    keywords='convert tool',
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
        'ait_sub_task': ait_sub_task_entry_points,
        'ait_sub_task_installer': ['ait-convert=model_convert.__install__:ConvertInstall'],
    },
)
