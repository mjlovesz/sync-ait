package com.ascend.ait.ide.commonlib.util.safeCmd;

import org.jetbrains.annotations.Nullable;

public abstract class CmdStr {
    /**
     * is the cmd safe
     *
     * @return is safe
     */
    public abstract boolean isSafe();

    /**
     * get last unsafe param
     *
     * @return last unsafe param
     */
    @Nullable
    public abstract String getLastUnSafeParam();
}
