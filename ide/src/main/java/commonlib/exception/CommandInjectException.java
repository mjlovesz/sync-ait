/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

package com.ascend.ait.ide.commonlib.exception;


import com.ascend.ait.ide.commonlib.util.BundleUtil;

/**
 * mindstudio - CommandInjectException
 *
 * @author liucj
 * @since 2021/4/14
 */
public class CommandInjectException extends Exception {
    private static final long serialVersionUID = -2439139320983098242L;

    public CommandInjectException() {
        this("");
    }

    public CommandInjectException(String errorParam) {
        super(BundleUtil.getCommonlibsString("command.inject.error") + "The error param is " + errorParam);
    }
}
