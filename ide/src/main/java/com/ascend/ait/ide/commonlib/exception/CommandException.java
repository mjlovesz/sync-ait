package com.ascend.ait.ide.commonlib.exception;

public class CommandException extends RuntimeException {
    private static final long serialVersionUID = -7848065374983831216L;

    public CommandException(String message) {
        super(message);
    }

    public CommandException() {
        this("");
    }
}
