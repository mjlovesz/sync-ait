package com.huawei.ascend.ait.ide.util.exception;

import java.io.IOException;
import java.io.Serial;

public class ModelFileInvalidException extends IOException {
    @Serial
    private static final long serialVersionUID = 1410259294426109220L;

    public ModelFileInvalidException(String message) {
        super(message);
    }
}
