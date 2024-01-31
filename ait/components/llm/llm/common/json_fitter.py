# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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
import sys
import json
import base64

import onnx
from google.protobuf.json_format import Parse


def atbNodeToPlainNode(atbNodeDict, level, target_level):
    if target_level !=-1 and level >= target_level:
        return [atbNodeDict]
    
    # 递归元
    if "nodes" in atbNodeDict:
        plainNodes = []
        for nodeDict in atbNodeDict["nodes"]:
            plainNodes = plainNodes + atbNodeToPlainNode(nodeDict, level + 1, target_level)
        return plainNodes
    else:
        return [atbNodeDict]


def atbJsonDictNodeParse(atbJsonDict, target_level):
    plainAtbNodes = []
    if "nodes" in atbJsonDict:
        rawAtbNodes = atbJsonDict["nodes"]
        level = 0
        for node in rawAtbNodes:
            plainAtbNodes = plainAtbNodes + atbNodeToPlainNode(node, level, target_level)
        return plainAtbNodes
    else:
        return []


def atbParamToOnnxAttribute(atbParamName, atbParamValue):
    onnxAttrDict = {}
    onnxAttrDict["name"] = atbParamName

    if isinstance(atbParamValue, str):
        onnxAttrDict["type"] = "STRINGS"
        onnxAttrDict["strings"] = [str(base64.b64decode(atbParamValue.encode("utf-8")), "utf-8")]
        return onnxAttrDict

    onnxAttrDict["type"] = "FLOATS"
    values = []
    if isinstance(atbParamValue, list):
        for v in atbParamValue:
            values.append(float(v))
    else:
        values.append(float(atbParamValue))
    onnxAttrDict["floats"] = values
    return onnxAttrDict


def parseOnnxAttrFromAtbNodeDict(atbNodeDict):
    onnxAttrs = []

    if "param" not in atbNodeDict:
        return onnxAttrs
    
    for paramName in atbNodeDict["param"]:
        if isinstance(atbNodeDict["param"][paramName], dict):
            for subParamName in atbNodeDict["param"][paramName]:
                fullName = paramName + "." + subParamName
                onnxAttrDict = atbParamToOnnxAttribute(fullName, atbNodeDict["param"][paramName][subParamName])
                onnxAttrs.append(onnxAttrDict)
        else:
            onnxAttrDict = atbParamToOnnxAttribute(fullName, atbNodeDict["param"][paramName])
            onnxAttrs.append(onnxAttrDict)
    return onnxAttrs


def atbNodeToOnnxNode(atbNodeDict):
    onnxNodeDict = {}
    onnxNodeDict["name"] = atbNodeDict["opName"]
    onnxNodeDict["opType"] = atbNodeDict["opType"]
    onnxNodeDict["input"] = atbNodeDict["inTensors"]
    onnxNodeDict["output"] = atbNodeDict["outTensors"]
    onnxNodeDict["attribute"] = parseOnnxAttrFromAtbNodeDict(atbNodeDict)
    return onnxNodeDict


def atbJsonToOnnxJson(atbJsonDict, target_level):
    onnxJsonDict = {}
    plain_nodes = atbJsonDictNodeParse(atbJsonDict, target_level)

    for i in range(len(plain_nodes)):
        plain_nodes[i] = atbNodeToOnnxNode(plain_nodes[i])

    onnxJsonDict["graph"] = {}
    onnxJsonDict["graph"] ["nodes"] = plain_nodes

    onnxJsonDict["graph"] ["input"] = []
    for inTensorName in atbJsonDict["inTensors"]:
        onnxInputTensorDict = {}
        onnxInputTensorDict["name"] = inTensorName
        onnxJsonDict["graph"]["input"].append(onnxInputTensorDict)

    onnxJsonDict["graph"] ["output"] = []
    for outTensorName in atbJsonDict["outTensors"]:
        onnxOutputTensorDict = {}
        onnxOutputTensorDict["name"] = outTensorName
        onnxJsonDict["graph"]["output"].append(onnxOutputTensorDict)
    return onnxJsonDict


def atbJsonToOnnx(atbJsonPath, target_level=-1):
    with open(atbJsonPath, "r") as file:
        jsonContent = json.loads(file, parse_constant=lambda x: None)
        onnxJson = atbJsonToOnnxJson(jsonContent, target_level)
        onnxStr = json.dumps(onnxJson)
        convertModel = Parse(onnxStr, onnx.ModelProto())
        onnxDir = atbJsonPath[0:-5] + ".onnx"
        onnx.save(convertModel, onnxDir)
