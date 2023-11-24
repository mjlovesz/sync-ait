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

import static com.huawei.ascend.ait.ide.commonlib.util.safecmd.CmdStrWordStatic.SPACE;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.huawei.ascend.ait.ide.commonlib.util.safecmd.CmdStrBuffer;
import com.huawei.ascend.ait.ide.commonlib.util.safecmd.CmdStrWordStatic;

import org.junit.jupiter.api.Test;

class AisBenchCmdStrTest {

    @Test
    void add() {
        CmdStrBuffer buffer = new CmdStrBuffer();
        buffer.append("param1").append(SPACE).append("param2").append(SPACE);
        CmdStrBuffer strBuffer = new CmdStrBuffer();
        AisBenchCmdStr.add(strBuffer, "param1", "param2");
        assertEquals(buffer.toString(), strBuffer.toString());

        CmdStrBuffer strBuffer2 = new CmdStrBuffer();
        AisBenchCmdStr.add(strBuffer2, "param1", "");
        assertEquals(new CmdStrBuffer().toString(), strBuffer2.toString());
    }

    @Test
    void addString() {
        CmdStrBuffer buffer = new CmdStrBuffer();
        buffer.append("param1").append(SPACE).append(new CmdStrWordStatic("param2")).append(SPACE);
        CmdStrBuffer strBuffer = new CmdStrBuffer();
        AisBenchCmdStr.addString(strBuffer, "param1", "param2");
        assertEquals(buffer.toString(), strBuffer.toString());

        CmdStrBuffer strBuffer2 = new CmdStrBuffer();
        AisBenchCmdStr.addString(strBuffer2, "param1", "");
        assertEquals(new CmdStrBuffer().toString(), strBuffer2.toString());
    }

    @Test
    void addState() {
        CmdStrBuffer buffer = new CmdStrBuffer();
        buffer.append("param1").append(SPACE).append(new CmdStrWordStatic("True")).append(SPACE);
        CmdStrBuffer strBuffer = new CmdStrBuffer();
        AisBenchCmdStr.addState(strBuffer, "param1", true);
        assertEquals(buffer.toString(), strBuffer.toString());

        CmdStrBuffer strBuffer2 = new CmdStrBuffer();
        strBuffer2.append("param1").append(SPACE).append(new CmdStrWordStatic("False")).append(SPACE);
        AisBenchCmdStr.addState(strBuffer2, "param1", false);
        assertEquals(strBuffer2.toString(), strBuffer2.toString());
    }
}
