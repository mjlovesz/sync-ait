# Copyright 2022 Huawei Technologies Co., Ltd
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
import click

@click.command(
    'list',
    short_help='List available Knowledges.[可用知识库列表]')
def command_list() -> None:
    print('Available knowledges: XXXXX')

@click.command(
    'evaluate',
    short_help='Evaluate model matching specified knowledges.'
)
def evaluate() -> None:
    print('command_evaluate')


surgeon_cmd_group = click.Group(name="surgeon", commands=[command_list, evaluate], help="使能ONNX模型在昇腾芯片的优化，并提供基于ONNX的改图功能")



