package com.ascend.ait.ide.commonlib.exception;


import com.ascend.ait.ide.commonlib.util.BundleUtil;

public class CommandInjectException extends Exception {
    private static final long serialVersionUID = -2439139320983098242L;

    public CommandInjectException() {
        this("");
    }

    public CommandInjectException(String errorParam) {
        super(BundleUtil.getCommonlibsString("command.inject.error") + "The error param is " + errorParam);
    }
}
