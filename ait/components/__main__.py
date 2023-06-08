# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import click
import pkg_resources

from components.debug import debug_cli_group
from components.profile import profile_cli
from components.transplt import transplt_cli
from components.benchmark import benchmark_cli
from components.analyze import analyze_cli
from components.convert import convert_cli


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


cli = click.Group(context_settings=CONTEXT_SETTINGS,
                  commands=[debug_cli_group, profile_cli,
                            analyze_cli, benchmark_cli, 
                            transplt_cli, convert_cli],
                  no_args_is_help=True,
                  help="ait(Ascend Inference Tools), "
                  "provides one-site debugging and optimization toolkit for inference use Ascend Devices")

if __name__ == "__main__":
    cli()
