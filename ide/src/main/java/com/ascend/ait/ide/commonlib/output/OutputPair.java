package com.ascend.ait.ide.commonlib.output;

import com.intellij.execution.ui.ConsoleViewContentType;

public class OutputPair {
    private String text;

    private ConsoleViewContentType contentType;

    private OutputType outputType;

    public OutputPair(String text, ConsoleViewContentType contentType) {
        this(text, OutputType.NORMAL, contentType);
    }

    public OutputPair(String text, OutputType outputType, ConsoleViewContentType contentType) {
        this.text = text;
        this.contentType = contentType;
        this.outputType = outputType;
    }

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }

    public ConsoleViewContentType getContentType() {
        return contentType;
    }

    public void setContentType(ConsoleViewContentType contentType) {
        this.contentType = contentType;
    }

    public OutputType getOutputType() {
        return outputType;
    }

    public void setOutputType(OutputType outputType) {
        this.outputType = outputType;
    }
}
