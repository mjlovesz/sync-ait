import os
copyright = '''# Copyright 2023 Huawei Technologies Co., Ltd
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
'''

def add_header(file, header):
    with open(file, 'r') as f:
        contents = f.read()
    contents = f"{header}\n{contents}"
    with open(file, 'w') as f:
        f.write(contents)

current_dir = os.getcwd()
for filename in os.listdir(current_dir):
    full_path = os.path.join(current_dir, filename)
    if filename.endswith('.py'):
        add_header(full_path, copyright)
    elif os.path.isdir(full_path):
        for file in os.listdir(full_path):
            file = os.path.join(full_path, file)
            if filename.endswith('.py'):
                add_header(full_path, copyright)


