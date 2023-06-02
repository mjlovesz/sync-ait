/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.ascend.ait.ide.service;

import com.ascend.ait.ide.commonlib.util.safeCmd.CmdStrBuffer;

public class AisBenchService {
    public static final String modelService = " --model ";
    public static final String inputService = " --input ";
    public static final String pureService = " --pure ";
    public static final String outputService = " --output ";
    public static final String outputdirService = " --outputdir ";
    public static final String outfmtService = " --outfmt ";
    public static final String loopService = " --loop ";
    public static final String warmupService = " --warmup_count ";
    public static final String deviceService = " --device ";
    public static final String debugService = " --debug ";
    public static final String displayService = " --display_all_summary ";

    public void pathAdd(CmdStrBuffer strBuffer, String service, String param) {
        if (!param.isEmpty()) {
            strBuffer.append(service).appendFilePath(param);
        }
    }

    public void strAdd(CmdStrBuffer strBuffer, String service, String param) {
        if (!param.isEmpty()) {
            strBuffer.append(service).append(param);
        }
    }

    public void statueAdd(CmdStrBuffer strBuffer, String service, boolean isOn) {
        if (!isOn) {
            strBuffer.append(service).append("false");
        } else {
            strBuffer.append(service).append("true");
        }
    }
}
