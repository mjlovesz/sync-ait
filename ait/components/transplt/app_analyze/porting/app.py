# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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

import click

from app_analyze.common.kit_config import KitConfig
from app_analyze.scan.scan_api import ScanApi


def start_scan_kit(args):
    if not os.path.exists(args.source):
        raise Exception("Source directory is not existed!")
    args.source = os.path.abspath(args.source)

    KitConfig.SOURCE_DIRECTORY = args.source
    scan_api = ScanApi()
    scan_api.scan_source(args)


def check_args(ctx: click.Context, params: click.Option, value: str):
    """
    check whether the param is provided
    """
    args = [
        opt
        for param in ctx.command.params
        for opt in param.opts
    ]
    if value in args:
        raise click.MissingParameter()
    return value


# 添加需要进行扫描的目录列表，逗号分隔的情况下只有一个列表元素
opt_source = click.option(
    '-s',
    '--source',
    'source',
    type=str,
    required=True,
    callback=check_args,
    help='directories of source folder'
)

# 添加输出格式选项，当前默认和仅支持xlsx
opt_report_type = click.option(
    '-f',
    '--report-type',
    'report_type',
    type=click.Choice(['csv', 'json']),
    default='csv',
    help='specify output report type. Only csv(xlsx)/json is supported.'
)

# 添加日志级别开关，默认级别是INFO，只有添加ERROR后才能输出ERR级别日志
opt_log_level = click.option(
    '-l',
    '--log-level',
    'log_level',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
    default='INFO',
    help='specify log level. default is '
         'INFO. choices from: DEBUG, INFO, WARNING, ERROR'
)

opt_tools = click.option(
    '-t',
    '--tools',
    'tools',
    type=click.Choice(KitConfig.VALID_CONSTRUCT_TOOLS),
    default='cmake',
    help='specify construction. default is cmake.'
)
