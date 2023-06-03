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

package com.huawei.ascend.ait.ide.commonlib.exception;

/**
 * CommandException
 *
 * @author cabbage
 * @date 2023/06/03
 */
public class CommandException extends RuntimeException {
    private static final long serialVersionUID = -7848065374983831216L;

    /**
     * CommandException
     *
     * @param message message
     */
    public CommandException(String message) {
        super(message);
    }
}
