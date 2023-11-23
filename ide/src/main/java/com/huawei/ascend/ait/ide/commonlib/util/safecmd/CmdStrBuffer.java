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

import org.apache.commons.lang3.StringUtils;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

public class CmdStrBuffer extends CmdStr {
    private static final String linuxLikeSep = "/";
    private static boolean isLinuxLikeSep = File.separator.equals(linuxLikeSep);
    private StringBuffer stringBuffer = new StringBuffer();
    private String lastUnSafeParam = null;
    private boolean isRemoteCmd = false;

    /**
     * check if it is safe param
     *
     * @param str param
     * @return is safe
     */
    public static boolean isSafeParam(@NotNull String str) {
        return Pattern.matches("[^|;&$><`\\\\!\n]*", str);
    }

    /**
     * simple make cmd buffer object
     *
     * @return cmd buffer
     */
    @NotNull
    public static CmdStrBuffer of() {
        return new CmdStrBuffer();
    }

    /**
     * simple make cmd buffer object
     *
     * @param param param
     * @return cmd buffer
     */
    @NotNull
    public static CmdStrBuffer of(@NotNull String param) {
        CmdStrBuffer buffer = new CmdStrBuffer();
        buffer.append(param);
        return buffer;
    }

    /**
     * simple make cmd buffer object
     *
     * @param param param
     * @return cmd buffer
     */
    @NotNull
    public static CmdStrBuffer of(@NotNull CmdStr param) {
        CmdStrBuffer buffer = new CmdStrBuffer();
        buffer.append(param);
        return buffer;
    }

    /**
     * append param, will check param
     *
     * @param param param
     * @return this, call chaining
     */
    @NotNull
    public CmdStrBuffer append(@NotNull String param) {
        if (isSafeParam(param)) {
            stringBuffer.append(param);
        } else {
            lastUnSafeParam = param;
        }
        return this;
    }

    /**
     * append cmd str, if append a bad cmd buffer will return false
     *
     * @param cmdBuffer cmd buffer
     * @return this, call chaining
     */
    @NotNull
    public CmdStrBuffer append(@NotNull CmdStr cmdBuffer) {
        if (cmdBuffer.isSafe()) {
            stringBuffer.append(cmdBuffer.toString());
        } else {
            lastUnSafeParam = cmdBuffer.getLastUnSafeParam();
        }
        return this;
    }

    /**
     * append file path
     *
     * @param pathStr path string
     * @return this, call chaining
     */
    @NotNull
    public CmdStrBuffer appendFilePath(@NotNull String pathStr) {
        if (isRemoteCmd && !isLinuxLikeSep) {
            return append(pathStr.replace(File.separator, linuxLikeSep));
        } else {
            if (!isLinuxLikeSep) {
                return appendJoinParam(Arrays.asList(StringUtils.split(pathStr, File.separatorChar)),
                        CmdStrWordStatic.FILE_SEP);
            } else {
                return append(pathStr);
            }
        }
    }

    /**
     * append join params ,like param1;param2;param3
     *
     * @param params format string
     * @param sep    params
     * @return this, call chaining
     */
    @NotNull
    public CmdStrBuffer appendJoinParam(@NotNull List<String> params, @NotNull CmdStr sep) {
        if (!sep.isSafe()) {
            lastUnSafeParam = sep.getLastUnSafeParam();
            return this;
        }
        for (String param : params) {
            if (!isSafeParam(param)) {
                lastUnSafeParam = param;
                return this;
            }
        }
        stringBuffer.append(String.join(sep.toString(), params));
        return this;
    }

    /**
     * append format string, like String.format
     *
     * @param formatString format string
     * @param params       params
     * @return this, call chaining
     */
    @NotNull
    public CmdStrBuffer appendFormat(@NotNull CmdStr formatString, @NotNull Object... params) {
        if (!formatString.isSafe()) {
            lastUnSafeParam = formatString.getLastUnSafeParam();
            return this;
        }
        for (Object param : params) {
            if (param instanceof CmdStr) {
                CmdStr paramBuffer = (CmdStr) param;
                if (!paramBuffer.isSafe()) {
                    lastUnSafeParam = paramBuffer.getLastUnSafeParam();
                    return this;
                }
            } else {
                if (!isSafeParam(param.toString())) {
                    lastUnSafeParam = param.toString();
                    return this;
                }
            }
        }
        stringBuffer.append(String.format(Locale.ROOT, formatString.toString(), params));
        return this;
    }

    /**
     * append format string, like String.format. make sure all params and formatString is safe
     *
     * @param formatString string ,format string
     * @param params       params
     * @return this, call chaining
     */
    @NotNull
    public CmdStrBuffer appendFormat(@NotNull String formatString, @NotNull Object... params) {
        return appendFormat(CmdStrBuffer.of(formatString), params);
    }

    @Override
    public boolean isSafe() {
        return lastUnSafeParam == null;
    }

    @Override
    public String getLastUnSafeParam() {
        return lastUnSafeParam;
    }

    /**
     * clear all
     */
    public void clear() {
        stringBuffer = new StringBuffer();
        lastUnSafeParam = null;
    }

    @Override
    public String toString() {
        return stringBuffer.toString();
    }

    /**
     * set remote cmd ,use by append path, set before append path
     *
     * @param isRemoteCmd remoteCmd
     * @return this, for call chain
     */
    public CmdStrBuffer setRemoteCmd(boolean isRemoteCmd) {
        this.isRemoteCmd = isRemoteCmd;
        return this;
    }
}
