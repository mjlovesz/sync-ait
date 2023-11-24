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

import org.jetbrains.annotations.NotNull;
import java.io.File;

public class CmdStrWordStatic extends CmdStr {
    // character used by the command

    /**
     * static word : COLON
     */
    public static final CmdStrWordStatic COLON = new CmdStrWordStatic(":");
    /**
     * static word : SEMICOLON
     */
    public static final CmdStrWordStatic SEMICOLON = new CmdStrWordStatic(";");
    /**
     * static word : AMPERSAND
     */
    public static final CmdStrWordStatic AMPERSAND = new CmdStrWordStatic("&");
    /**
     * static word : DOLLAR
     */
    public static final CmdStrWordStatic DOLLAR = new CmdStrWordStatic("$");
    /**
     * static word : GREATER_THAN
     */
    public static final CmdStrWordStatic GREATER_THAN = new CmdStrWordStatic(">");
    /**
     * static word : LESS_THAN
     */
    public static final CmdStrWordStatic LESS_THAN = new CmdStrWordStatic("<");
    /**
     * static word : BACK_QUOTE
     */
    public static final CmdStrWordStatic BACK_QUOTE = new CmdStrWordStatic("`");
    /**
     * static word : BACKSLASH
     */
    public static final CmdStrWordStatic BACKSLASH = new CmdStrWordStatic("\\");
    /**
     * static word : EXCLAMATION
     */
    public static final CmdStrWordStatic EXCLAMATION = new CmdStrWordStatic("!");
    /**
     * static word : SPACE
     */
    public static final CmdStrWordStatic SPACE = new CmdStrWordStatic(" ");
    /**
     * static word : SPACE
     */
    public static final CmdStrWordStatic FILE_SEP = new CmdStrWordStatic(File.separator);

    // Common format
    /**
     * static word : QUOTED_STRING
     */
    public static final CmdStrWordStatic QUOTED_STRING = new CmdStrWordStatic("\"%s\"");
    /**
     * static word : PARAM_KEY_VALUE
     */
    public static final CmdStrWordStatic PARAM_KEY_VALUE = new CmdStrWordStatic(" -%s %s ");
    /**
     * static word : PARAM_KEY_EQ_VALUE
     */
    public static final CmdStrWordStatic PARAM_KEY_EQ_VALUE = new CmdStrWordStatic(" --%s=%s ");

    private static final String UN_SAFE_PARAM = null;

    private String cmdStaticWord;

    public CmdStrWordStatic(@NotNull String cmdStaticWord) {
        this.cmdStaticWord = cmdStaticWord;
    }

    @Override
    public String toString() {
        return cmdStaticWord;
    }

    @Override
    public boolean isSafe() {
        return true;
    }

    @Override
    public String getLastUnSafeParam() {
        return UN_SAFE_PARAM;
    }
}
