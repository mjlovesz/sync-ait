# Copyright 2023 Huawei Technologies Co., Ltd
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
import pathlib
from typing import Dict, List

from src import analysis
from src.knowledge import Knowledge


def print_result(result: Dict[Knowledge, List[str]]):
    if len(result) == 0:
        return
    print()
    print('============= Analysis Result =============')
    print()
    for knowledge, match_infos in result.items():
        print(f'{knowledge._suggestion}')
        print('查询和匹配到的接口和路径如下：')
        for match_info in match_infos:
            print(f'  {match_info}')
        print()

opt_path = click.argument(
    'path',
    nargs=1,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        path_type=pathlib.Path
    )
)

opt_scene = click.option(
    '-s',
    '--scene',
    'scene',
    default='310->310B',
    type=str,
    help='scene you want to analysis, default 310->310B.'
)

@click.command()
@opt_path
@opt_scene
def analysis_acl_api(path, scene):
    """analysis application code and print suggestions
    """
    if scene != '310->310B':
        print(f'[error] not support scene: {scene}.')
        return

    result = analysis.analysis_310_to_310B(path)
    print_result(result)


if __name__ == '__main__':
    analysis_acl_api()
