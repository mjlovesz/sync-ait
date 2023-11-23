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

package com.huawei.ascend.ait.ide.commonlib.output;

import com.intellij.execution.ui.ConsoleViewContentType;

/**
 * OutputPair
 *
 * @author cabbage
 * @since 2023/06/03
 */
public class OutputPair {
    private String text;

    private ConsoleViewContentType contentType;

    private OutputType outputType;

    /**
     * OutputPair
     *
     * @param text      text
     * @param contentType  contentType
     */
    public OutputPair(String text, ConsoleViewContentType contentType) {
        this(text, OutputType.NORMAL, contentType);
    }

    /**
     * OutputPair
     *
     * @param text      text
     * @param outputType  outputType
     * @param contentType contentType
     */
    public OutputPair(String text, OutputType outputType, ConsoleViewContentType contentType) {
        this.text = text;
        this.contentType = contentType;
        this.outputType = outputType;
    }

    /**
     * getText
     *
     * @return String
     */
    public String getText() {
        return text;
    }

    /**
     * setText
     *
     * @param text text
     */
    public void setText(String text) {
        this.text = text;
    }

    /**
     * getContentType
     *
     * @return ConsoleViewContentType
     */
    public ConsoleViewContentType getContentType() {
        return contentType;
    }

    /**
     * setContentType
     *
     * @param contentType contentType
     */
    public void setContentType(ConsoleViewContentType contentType) {
        this.contentType = contentType;
    }

    /**
     * OutputType
     *
     * @return 0utputType
     */
    public OutputType getOutputType() {
        return outputType;
    }

    /**
     * setOutputType
     *
     * @param outputType outputType
     */
    public void setOutputType(OutputType outputType) {
        this.outputType = outputType;
    }
}
