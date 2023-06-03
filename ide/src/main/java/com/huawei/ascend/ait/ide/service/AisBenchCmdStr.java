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

package com.huawei.ascend.ait.ide.service;

import com.huawei.ascend.ait.ide.commonlib.util.safeCmd.CmdStr;
import com.huawei.ascend.ait.ide.commonlib.util.safeCmd.CmdStrWordStatic;

import java.util.List;

public class AisBenchCmdStr {
    public static final CmdStrWordStatic modelService = new CmdStrWordStatic("--model");
    public static final CmdStrWordStatic inputService = new CmdStrWordStatic("--input");
    public static final CmdStrWordStatic pureService = new CmdStrWordStatic("--pure");
    public static final CmdStrWordStatic outputService = new CmdStrWordStatic("--output");
    public static final CmdStrWordStatic outputdirService = new CmdStrWordStatic("--outputdir");
    public static final CmdStrWordStatic outfmtService = new CmdStrWordStatic("--outfmt");
    public static final CmdStrWordStatic loopService = new CmdStrWordStatic("--loop");
    public static final CmdStrWordStatic warmupService = new CmdStrWordStatic("--warmup_count");
    public static final CmdStrWordStatic deviceService = new CmdStrWordStatic("--device");
    public static final CmdStrWordStatic debugService = new CmdStrWordStatic("--debug");
    public static final CmdStrWordStatic displayService = new CmdStrWordStatic("--display_all_summary");
    public static final CmdStrWordStatic falseService = new CmdStrWordStatic("false");
    public static final CmdStrWordStatic trueService = new CmdStrWordStatic("true");

    public static void addPath(List<CmdStr> strBuffer, CmdStrWordStatic service, CmdStrWordStatic param) {
        if (!param.toString().isEmpty()) {
            strBuffer.add(service);
            strBuffer.add(param);
        }
    }

    public static void addState(List<CmdStr> strBuffer, CmdStrWordStatic service, boolean isOn) {
        strBuffer.add(service);
        if (!isOn) {
            strBuffer.add(falseService);
        } else {
            strBuffer.add(trueService);
        }
    }
}
