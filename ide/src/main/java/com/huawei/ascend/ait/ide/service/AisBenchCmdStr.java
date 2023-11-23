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

import com.huawei.ascend.ait.ide.commonlib.util.safecmd.CmdStrBuffer;
import com.huawei.ascend.ait.ide.commonlib.util.safecmd.CmdStrWordStatic;

import java.util.Objects;

/**
 * AisBenchCmdStr
 *
 * @author cabbage
 * @since 2023/06/03
 */
public class AisBenchCmdStr {
    public static final CmdStrWordStatic falseService = new CmdStrWordStatic("False");
    public static final CmdStrWordStatic trueService = new CmdStrWordStatic("True");

    /**
     * addPath
     *
     * @param strBuffer strBuffer
     * @param param param
     * @param param1 param1
     */
    public static void add(CmdStrBuffer strBuffer, String param, String param1) {
        if (!Objects.equals(param1, "")) {
            strBuffer.append(param).append(CmdStrWordStatic.SPACE)
                    .append(param1).append(CmdStrWordStatic.SPACE);
        }
    }

    public static void addString(CmdStrBuffer strBuffer, String param, String param1) {
        if (!Objects.equals(param1, "")) {
            strBuffer.append(param).append(CmdStrWordStatic.SPACE)
                    .append(new CmdStrWordStatic(param1)).append(CmdStrWordStatic.SPACE);
        }
    }

    /**
     * addState
     *
     * @param strBuffer strBuffer
     * @param param param
     * @param isOn isOn
     */
    public static void addState(CmdStrBuffer strBuffer, String param, boolean isOn) {
        strBuffer.append(param).append(CmdStrWordStatic.SPACE);
        if (!isOn) {
            strBuffer.append(falseService).append(CmdStrWordStatic.SPACE);
        } else {
            strBuffer.append(trueService).append(CmdStrWordStatic.SPACE);
        }
    }
}
