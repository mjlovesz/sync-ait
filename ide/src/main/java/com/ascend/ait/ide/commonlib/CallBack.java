package com.ascend.ait.ide.commonlib;

public interface CallBack {
    void onInputMessage(String var1, String var2);

    void onErrorMessage(String var1, String var2);

    void onStageMessage(String var1, String var2);
}
