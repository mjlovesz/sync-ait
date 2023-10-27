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

package com.huawei.ascend.ait.ide.commonlib.util.safecmd;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.huawei.ascend.ait.ide.commonlib.exception.CommandInjectException;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

/**
 * CmdExec TEST
 *
 * @author admin
 * @since 2023/06/28
 */
class CmdExecTest {
    static final CmdStrWordStatic AND = new CmdStrWordStatic("&&");
    static final CmdStrBuffer strBuffer =  new CmdStrBuffer().append("echo 1").append(AND).append("echo 2");

    @BeforeEach
    void setUp() {
    }

    @AfterEach
    void tearDown() {
    }

    @Test
    void bashStart() {
        CmdExec exec = new CmdExec();
        CmdStrBuffer buffer = new CmdStrBuffer();
        buffer.append("echo hello world");
        assertDoesNotThrow(() -> {
            assertTrue(exec.bashStart(buffer));
        });

        assertEquals("hello world", exec.getResult());
    }

    @Test
    void bashStartLineSeq() {
        CmdExec exec = new CmdExec();
        CmdStrBuffer buffer = new CmdStrBuffer();
        buffer.append("echo hello world");
        buffer.append(AND).append("echo hello world");
        assertDoesNotThrow(() -> {
            assertTrue(exec.bashStart(buffer));
        });

        String exe1 = exec.getResult();
        String exe2 = "hello world" + System.lineSeparator() + "hello world";

        assertEquals(exe1, exe2);
    }

    @Test
    void startThrow() {
        CmdExec exec = new CmdExec();
        assertThrows(CommandInjectException.class, () -> {
            exec.pythonStart(CmdStrBuffer.of("-c"), CmdStrBuffer.of().
                    appendFormat(new CmdStrWordStatic("print(%s)"), "1$2$3"));
        });
    }

    @Test
    void setListener() {
        CmdExec exec = new CmdExec();
        List<String> all = new ArrayList();
        assertDoesNotThrow(() -> {
            exec.setListener(
                    (eventType, msg) -> {
                        if (msg != null) {
                            all.add(msg);
                        }
                    });
            exec.bashStart(strBuffer);
        });
        assertEquals(2, all.size());
    }

    @Test
    void getResult() {
        CmdExec exec = new CmdExec();
        CmdStrBuffer buffer =  new CmdStrBuffer().append("echo 1");
        assertDoesNotThrow(() -> {
            assertTrue(exec.bashStart(buffer));
        });

        assertEquals("1", exec.getResult());
    }

    @Test
    void getErrorResult() {
        CmdExec exec = new CmdExec();
        CmdStrBuffer buffer = new CmdStrBuffer();
        buffer.append("123");
        assertDoesNotThrow(() -> {
            assertFalse(exec.bashStart(buffer));
        });

        assertNotNull(exec.getErrorResult());
    }
}