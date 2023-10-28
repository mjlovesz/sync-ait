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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

/**
 * CmdStrBuffer Test
 *
 * @author cabbage
 * @since 2023/06/28
 */
class CmdStrBufferTest {
    private String goodStr = "abed";
    private String badStr = "abed; rm -rf some-path";

    @Test
    void isSafeParam() {
        //[^|;&$><`\\!\n]+
        assertFalse(CmdStrBuffer.isSafeParam("!"));
        assertFalse(CmdStrBuffer.isSafeParam("abed!abed"));
        assertTrue(CmdStrBuffer.isSafeParam("abed_rtz_abed"));
        assertTrue(CmdStrBuffer.isSafeParam(""));

        assertFalse(CmdStrBuffer.isSafeParam("abed|abed"));
        assertFalse(CmdStrBuffer.isSafeParam("abed; rm -rf some-path"));
        assertFalse(CmdStrBuffer.isSafeParam("abed& rm -rf  some-path"));
        assertFalse(CmdStrBuffer.isSafeParam("rm $HOME/some-file"));
        assertFalse(CmdStrBuffer.isSafeParam("echo hello > some-file"));
        assertFalse(CmdStrBuffer.isSafeParam("echo hello < some-file"));
        assertFalse(CmdStrBuffer.isSafeParam("abed\\n rm -rf some-path"));
        assertFalse(CmdStrBuffer.isSafeParam("abed\n rm -rf some-path"));
        assertFalse(CmdStrBuffer.isSafeParam("echo `rm -rf some-path`"));

        assertTrue(CmdStrBuffer.isSafeParam("n"));
        assertTrue(CmdStrBuffer.isSafeParam("^"));
        assertTrue(CmdStrBuffer.isSafeParam("["));
        assertTrue(CmdStrBuffer.isSafeParam("]"));
        assertTrue(CmdStrBuffer.isSafeParam("+"));
    }

    @Test
    void append() {
        CmdStrBuffer buffer = new CmdStrBuffer();
        buffer.append(goodStr);
        assertTrue(buffer.isSafe());
        buffer.append(badStr);
        assertFalse(buffer.isSafe());
        assertEquals(goodStr, buffer.toString());
        assertEquals(badStr, buffer.getLastUnSafeParam());
    }

    @Test
    void appendStatic() {
        CmdStrBuffer buffer = new CmdStrBuffer();
        buffer.append(CmdStrWordStatic.SEMICOLON);
        assertEquals(CmdStrWordStatic.SEMICOLON.toString(), buffer.toString());
    }

    @Test
    void appendCmdBuffer() {
        CmdStrBuffer bufferGood = new CmdStrBuffer();
        bufferGood.append(goodStr);
        CmdStrBuffer bufferBad = new CmdStrBuffer();
        bufferBad.append(badStr);

        CmdStrBuffer buffer = new CmdStrBuffer();
        buffer.append(bufferGood);
        assertTrue(buffer.isSafe());
        assertEquals(goodStr, buffer.toString());
        buffer.append(bufferBad);
        assertFalse(buffer.isSafe());
        assertEquals(goodStr, buffer.toString());
        assertEquals(badStr, buffer.getLastUnSafeParam());
    }

    @Test
    void AppendStaticParam() {
        CmdStrBuffer buffer = new CmdStrBuffer();
        buffer.append(CmdStrWordStatic.SEMICOLON).append(goodStr);
        assertTrue(buffer.isSafe());
        CmdStrBuffer bufferGood = new CmdStrBuffer();
        bufferGood.append(buffer).append(CmdStrWordStatic.SEMICOLON);
        buffer.append(CmdStrWordStatic.SEMICOLON).append(badStr);
        assertFalse(buffer.isSafe());
        assertEquals(badStr, buffer.getLastUnSafeParam());
        assertEquals(bufferGood.toString(), buffer.toString());
    }

    @Test
    void AppendParamStatic() {
        CmdStrBuffer buffer = new CmdStrBuffer();
        buffer.append(goodStr).append(CmdStrWordStatic.SEMICOLON);
        assertTrue(buffer.isSafe());
        CmdStrBuffer bufferGood = new CmdStrBuffer();
        bufferGood.append(buffer).append(CmdStrWordStatic.SEMICOLON);
        buffer.append(badStr).append(CmdStrWordStatic.SEMICOLON);
        assertFalse(buffer.isSafe());
        assertEquals(badStr, buffer.getLastUnSafeParam());
        assertEquals(bufferGood.toString(), buffer.toString());
    }

    @Test
    void appendFormat() {
        CmdStrBuffer buffer = new CmdStrBuffer();
        buffer.appendFormat(CmdStrWordStatic.QUOTED_STRING, goodStr);
        assertTrue(buffer.isSafe());
        buffer.appendFormat(CmdStrWordStatic.QUOTED_STRING, goodStr);
        assertTrue(buffer.isSafe());
        assertEquals(String.format(CmdStrWordStatic.QUOTED_STRING.toString(), goodStr) +
                String.format(CmdStrWordStatic.QUOTED_STRING.toString(), goodStr), buffer.toString());
        buffer.appendFormat(CmdStrWordStatic.QUOTED_STRING, badStr);
        assertFalse(buffer.isSafe());
    }

    @Test
    void isSafe() {
        CmdStrBuffer buffer = new CmdStrBuffer();
        buffer.appendFormat(CmdStrWordStatic.QUOTED_STRING, goodStr);
        assertTrue(buffer.isSafe());
        buffer.appendFormat(CmdStrWordStatic.QUOTED_STRING, badStr);
        assertFalse(buffer.isSafe());
    }

    @Test
    void getLastUnSafeParam() {
        CmdStrBuffer buffer = new CmdStrBuffer();
        buffer.appendFormat(CmdStrWordStatic.QUOTED_STRING, badStr);
        assertEquals(badStr, buffer.getLastUnSafeParam());
    }

    @Test
    void testToString() {
        CmdStrBuffer buffer = new CmdStrBuffer();
        buffer.append(goodStr);
        assertEquals(goodStr, buffer.toString());
    }

    @Test
    void clear() {
        CmdStrBuffer buffer = new CmdStrBuffer();
        buffer.append(goodStr);
        buffer.append(badStr);
        buffer.clear();
        assertTrue(buffer.isSafe());
        assertEquals("", buffer.toString());
    }

    @Test
    void appendJoinParam() {
        List<String> list = Arrays.asList("a","b","c");
        assertEquals("start=a;b;c", CmdStrBuffer.of("start=")
                .appendJoinParam(list, CmdStrWordStatic.SEMICOLON).toString());

        assertFalse(CmdStrBuffer.of("start=").appendJoinParam(list, CmdStrBuffer.of(";")).isSafe());

        List<String> list2 = Arrays.asList("a","b","c","&");
        assertFalse(CmdStrBuffer.of("start=").appendJoinParam(list2, CmdStrWordStatic.SEMICOLON).isSafe());
    }
}