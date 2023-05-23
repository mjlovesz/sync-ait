package com.ascend.ait.ide.util;

import com.huawei.mindstudio.annotations.NotNull;

public interface ais_service {
    String getServiceName();
    Integer getServiceCode();
    @NotNull
    ExceptionBean getCommand();
    boolean isAvaliable();

}
