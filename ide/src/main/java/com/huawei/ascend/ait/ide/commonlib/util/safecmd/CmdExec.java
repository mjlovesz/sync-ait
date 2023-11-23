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

package com.huawei.ascend.ait.ide.commonlib.util.safecmd;

import com.huawei.ascend.ait.ide.commonlib.exception.CommandInjectException;
import org.apache.commons.lang3.SystemUtils;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class CmdExec {
    /**
     * exec event type
     */
    public enum EventType {
        INFO_OUTPUT_MSG,
        ERROR_OUTPUT_MSG,
        USER_CANCEL,
        END_CANCEL,
        END_TIMEOUT,
        FINISH,
        POLLING
    }

    /**
     * command result listener
     */
    public interface ResultListener {
        /**
         * result callback function
         *
         * @param msg       callback message
         * @param eventType event type
         */
        void onResult(@NotNull EventType eventType, @Nullable String msg);
    }

    /**
     * cancel status, Polling status
     */
    public interface StopCallback {
        /**
         * get cancel status
         *
         * @return is cancel
         */
        boolean isStop();
    }

    // will delete when CommandUtils changed
    private static final CmdStrWordStatic BASH = new CmdStrWordStatic("/bin/bash");
    private static final CmdStrWordStatic PARAM_C = new CmdStrWordStatic("-c");
    private static final CmdStrWordStatic PYTHON = new CmdStrWordStatic("python3.7");
    private static final CmdStrWordStatic BASH_WIN = new CmdStrWordStatic("cmd");
    private static final CmdStrWordStatic PARAM_C_WIN = new CmdStrWordStatic("/c");
    private static final CmdStrWordStatic PYTHON_WIN = new CmdStrWordStatic("python3");

    private Process runProcess;
    private boolean canceled = false;
    private StringBuffer result;
    private StringBuffer errorResult;
    private StopCallback stopCallback;
    private boolean enablePollingEvent = false;
    private int timeout = Integer.MAX_VALUE; // default 60 years
    private boolean timedOut = false;
    private ResultListener listener = new ResultListener() {
        @Override
        public void onResult(@NotNull EventType eventType, String msg) {
            switch (eventType) {
                case INFO_OUTPUT_MSG:
                    result = appendMsg(result, msg);
                    break;
                case ERROR_OUTPUT_MSG:
                    errorResult = appendMsg(errorResult, msg);
                    break;
                default:
                    break;
            }
        }

        private StringBuffer appendMsg(StringBuffer ori, String msg) {
            if (msg == null) {
                return ori;
            } else if (ori == null) {
                return new StringBuffer(msg);
            } else {
                return ori.append(System.lineSeparator()).append(msg);
            }
        }
    };

    /**
     * bash run simple str command, whole str will check cmd inject
     *
     * @param cmdStr str to run
     * @return is success
     * @throws CommandInjectException inject exception
     * @throws IOException            io exception
     */
    public static boolean bashExec(@NotNull String cmdStr) throws CommandInjectException, IOException {
        return bashExec(CmdStrBuffer.of(cmdStr));
    }

    /**
     * bash run simple command
     *
     * @param cmdStr str to run
     * @return is success
     * @throws CommandInjectException inject exception
     * @throws IOException            io exception
     */
    public static boolean bashExec(@NotNull CmdStr cmdStr) throws CommandInjectException, IOException {
        CmdExec exec = new CmdExec();
        return exec.bashStart(cmdStr);
    }

    /**
     * python run file and with param
     *
     * @param cmdStr file and param of python
     * @return is success
     * @throws CommandInjectException inject exception
     * @throws IOException            io exception
     */
    public static boolean pythonExec(@NotNull CmdStr... cmdStr) throws CommandInjectException, IOException {
        CmdExec exec = new CmdExec();
        return exec.pythonStart(cmdStr);
    }

    /**
     * start bash simple command, after like set listener
     *
     * @param cmdStr str to run
     * @return is success
     * @throws CommandInjectException inject exception
     * @throws IOException            io exception
     */
    public boolean bashStart(@NotNull CmdStr cmdStr) throws CommandInjectException, IOException {
        if (SystemUtils.IS_OS_WINDOWS) {
            return start(BASH_WIN, PARAM_C_WIN, cmdStr);
        } else {
            return start(BASH, PARAM_C, cmdStr);
        }
    }

    /**
     * start python progress, after like set listener
     *
     * @param cmdStr str to run
     * @return is success
     * @throws CommandInjectException inject exception
     * @throws IOException            io exception
     */
    public boolean pythonStart(@NotNull CmdStr... cmdStr) throws CommandInjectException, IOException {
        if (SystemUtils.IS_OS_WINDOWS) {
            return start(PYTHON_WIN, cmdStr);
        } else {
            return start(PYTHON, cmdStr);
        }
    }

    /**
     * start run
     *
     * @param cmd       cmd
     * @param cmdParams param
     * @return is success
     * @throws CommandInjectException inject exception
     * @throws IOException            io exception
     */
    public boolean start(@NotNull CmdStrWordStatic cmd, @NotNull CmdStr... cmdParams)
            throws CommandInjectException, IOException {
        initStatus();
        List<String> commands = new ArrayList<>();
        commands.add(cmd.toString());

        for (CmdStr cmdParam : cmdParams) {
            if (!cmdParam.isSafe()) {
                throw new CommandInjectException(cmdParam.getLastUnSafeParam());
            }
            commands.add(cmdParam.toString());
        }

        ProcessBuilder pb = new ProcessBuilder();
        pb.redirectErrorStream(false);
        pb.command(commands);

        synchronized (this) {
            if (canceled) {
                return false;
            }
            runProcess = pb.start();
        }

        try {
            getProgressOutput(runProcess);
            if (isCanceled()) {
                listener.onResult(EventType.END_CANCEL, null);
                return false;
            } else if (timedOut) {
                listener.onResult(EventType.END_TIMEOUT, null);
                return false;
            } else {
                listener.onResult(EventType.FINISH, null);
                return runProcess.waitFor() == 0;
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    /**
     * is canceled
     *
     * @return is
     */
    public boolean isCanceled() {
        return canceled;
    }

    /**
     * stop progress, like user click cancel. cmd always run in work thread ,and stop in UI thread
     * if start run in UI thread, it takes no effect
     */
    public void stop() {
        synchronized (this) {
            if (canceled) {
                return;
            }
            canceled = true;
            if (runProcess != null && runProcess.isAlive()) {
                runProcess.destroy();
            }
            if (runProcess == null || runProcess.isAlive()) {
                if (listener != null) {
                    listener.onResult(EventType.USER_CANCEL, null);
                }
            }
        }
    }

    /**
     * provide a callback to get cancel status.
     *
     * @param stopCallBack stop callback
     */
    public void setStopCallback(StopCallback stopCallBack) {
        this.stopCallback = stopCallBack;
    }

    /**
     * set listener, set before start, otherwise unknown error
     * when set listener, the function get result and get error result will return null
     *
     * @param listener call back
     */
    public void setListener(@NotNull ResultListener listener) {
        this.listener = listener;
    }

    /**
     * set enable nothing event, each loop get a nothing event in polling
     * for status check or other use
     *
     * @param enablePollingEvent enable
     */
    public void setEnablePollingEvent(boolean enablePollingEvent) {
        this.enablePollingEvent = enablePollingEvent;
    }

    /**
     * set timeout ,unit is sec
     * default 60+ years
     *
     * @param timeout timeout
     */
    public void setTimeout(int timeout) {
        this.timeout = timeout;
    }

    /**
     * get result string, when set listener, the function will return null
     *
     * @return result
     */
    public String getResult() {
        String resultString = null;
        if (result != null) {
            resultString = result.toString();
        }
        return resultString;
    }

    /**
     * get error result string, when set listener, the function will return null
     *
     * @return error result
     */
    public String getErrorResult() {
        String errorString = null;
        if (errorResult != null) {
            errorString = errorResult.toString();
        }
        return errorString;
    }

    private void getProgressOutput(Process process) throws IOException, InterruptedException {
        if (process == null) {
            return;
        }
        long startTime = getTime();

        try (InputStreamReader inputStreamReader =
                new InputStreamReader(process.getInputStream(), Charset.defaultCharset());
                BufferedReader inputBR = new BufferedReader(inputStreamReader);
                InputStreamReader errorStreamReader =
                        new InputStreamReader(process.getErrorStream(), Charset.defaultCharset());
                BufferedReader errorBR = new BufferedReader(errorStreamReader)) {
            String inputLine;
            String errorLine;
            while ((process.isAlive() || errorBR.ready() || inputBR.ready())) {
                boolean gotSomething = false;
                if (errorBR.ready()) {
                    gotSomething = true;
                    errorLine = errorBR.readLine();
                    if (listener != null) {
                        listener.onResult(EventType.ERROR_OUTPUT_MSG, errorLine);
                    }
                }

                if (inputBR.ready()) {
                    gotSomething = true;
                    inputLine = inputBR.readLine();
                    if (listener != null) {
                        listener.onResult(EventType.INFO_OUTPUT_MSG, inputLine);
                    }
                }

                if (stopCallback != null && !canceled && stopCallback.isStop()) {
                    stop();
                }

                if (enablePollingEvent && listener != null) {
                    listener.onResult(EventType.POLLING, null);
                }

                if (getTime() - startTime > timeout) {
                    stop();
                    timedOut = true;
                }

                if (!gotSomething) {
                    Thread.sleep(15);
                }
            }
        }
    }

    private void initStatus() {
        runProcess = null;
        canceled = false;
        result = null;
        errorResult = null;
    }

    private long getTime() {
        return new Date().getTime() / 1000;
    }
}
