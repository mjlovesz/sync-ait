package com.huawei.ascend.ait.ide.util.exception;

import java.io.IOException;
import java.io.Serial;

public class PathInvalidException extends IOException {
    @Serial
    private static final long serialVersionUID = -7964383566956264133L;

    public PathInvalidException(String message) {
        super(message);
    }
}
