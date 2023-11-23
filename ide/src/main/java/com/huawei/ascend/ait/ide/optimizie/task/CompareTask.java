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

package com.huawei.ascend.ait.ide.optimizie.task;

import com.huawei.ascend.ait.ide.commonlib.exception.CommandInjectException;
import com.huawei.ascend.ait.ide.commonlib.output.OutputService;
import com.huawei.ascend.ait.ide.commonlib.ui.SwitchButton;
import com.huawei.ascend.ait.ide.commonlib.util.safecmd.CmdExec;
import com.huawei.ascend.ait.ide.commonlib.util.safecmd.CmdStrBuffer;
import com.huawei.ascend.ait.ide.optimizie.ui.step.ShowResult;

import com.intellij.execution.ui.ConsoleViewContentType;
import com.intellij.openapi.application.ApplicationManager;
import com.intellij.openapi.progress.ProgressIndicator;
import com.intellij.openapi.progress.Task.Backgroundable;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.project.ProjectManager;
import com.intellij.openapi.project.ProjectManagerListener;

import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

/**
 * CompareTask
 *
 * @author cabbage
 * @since 2023/06/25
 */
public class CompareTask extends Backgroundable {
    private static final Logger LOGGER = LoggerFactory.getLogger(CompareTask.class);
    private final Project project;
    private CmdExec exec;
    private CmdStrBuffer cmdStrBuffer;
    private SwitchButton dump;
    private String csvPath;
    private final Object lock = new Object();

    /**
     * CompareTask
     *
     * @param project          project
     * @param cmdStrBuffer     cmdStrBuffer
     * @param dumpButton       dumpButton
     */
    public CompareTask(@NotNull Project project, CmdStrBuffer cmdStrBuffer, SwitchButton dumpButton) {
        super(project, "Compare ...", true);
        this.project = project;
        this.cmdStrBuffer = cmdStrBuffer;
        this.dump = dumpButton;
        ProjectManager.getInstance().addProjectManagerListener(project, new ProjectManagerListener() {
            @Override
            public void projectClosing(@NotNull Project project) {
                stopCompare();
            }
        });
        OutputService.getInstance(project).clear();
        OutputService.getInstance(project).active();
        OutputService.getInstance(project).show();
    }

    /**
     * stopCompare
     */
    public void stopCompare() {
        synchronized (lock) {
            if (exec == null || exec.isCanceled()) {
                return;
            }
            exec.stop();
        }
    }

    private void executeThreadFunction(@NotNull ProgressIndicator indicator) {
        while (indicator.isRunning()) {
            try {
                if (indicator.isCanceled()) {
                    OutputService.getInstance(project).print("Compare has been stopped by the user.",
                            ConsoleViewContentType.LOG_ERROR_OUTPUT);
                    stopCompare();
                    return;
                }
                Thread.sleep(20);
            } catch (InterruptedException exception) {
                throw new RuntimeException(exception);
            }
        }
    }

    private void executeThread(@NotNull ProgressIndicator indicator) {
        ApplicationManager.getApplication().executeOnPooledThread(() -> {
            executeThreadFunction(indicator);
        });
    }

    /**
     * doCompare
     *
     * @param indicator indicator
     * @return boolean
     */
    private boolean doCompare(@NotNull ProgressIndicator indicator) {
        exec = new CmdExec();

        boolean compareSuccess = false;
        OutputService.getInstance(project).print("Start compare.", ConsoleViewContentType.LOG_INFO_OUTPUT);
        OutputService.getInstance(project).print("Please wait for a moment:", ConsoleViewContentType.LOG_INFO_OUTPUT);
        CmdStrBuffer CompareCmd = cmdStrBuffer;
        OutputService.getInstance(project).print(CompareCmd.toString());
        try {
            exec.setStopCallback(indicator::isCanceled);
            exec.setListener(cmdResultListener);
            if (exec.bashStart(CompareCmd)) {
                compareSuccess = true;
            }
        } catch (CommandInjectException | IOException exception) {
            LOGGER.error(exception.getMessage());
        }
        return compareSuccess;
    }

    private void compareSuccess() {
        OutputService.getInstance(project).print("Compare successfully.", ConsoleViewContentType.LOG_INFO_OUTPUT);
        showCsv();
    }

    private void compareFail() {
        OutputService.getInstance(project).print("Compare unsuccessfully.", ConsoleViewContentType.LOG_ERROR_OUTPUT);
    }

    /**
     * run
     *
     * @param indicator indicator
     */

    @Override
    public void run(@NotNull ProgressIndicator indicator) {
        OutputService.getInstance(project).scrollToEnd();
        executeThread(indicator);

        boolean isCompareSuccess = doCompare(indicator);
        if (isCompareSuccess) {
            compareSuccess();
        } else {
            compareFail();
        }
    }

    private final CmdExec.ResultListener cmdResultListener = new CmdExec.ResultListener() {
        @Override
        public void onResult(@NotNull CmdExec.EventType eventType, String msg) {
            if (msg == null) {
                return;
            }
            if (msg.contains("csv_path=")) {
                String[] path = msg.split("csv_path=");
                if (path.length >= 2) {
                    csvPath = path[1];
                }
            }
            if (msg.contains("INFO")) {
                OutputService.getInstance(project).print(msg, ConsoleViewContentType.LOG_INFO_OUTPUT);
            } else if (msg.contains("WARNING")) {
                OutputService.getInstance(project).print(msg, ConsoleViewContentType.LOG_WARNING_OUTPUT);
            } else if (msg.contains("ERROR")) {
                OutputService.getInstance(project).print(msg, ConsoleViewContentType.LOG_ERROR_OUTPUT);
            } else {
                OutputService.getInstance(project).print(msg, ConsoleViewContentType.SYSTEM_OUTPUT);
            }
        }
    };

    private void showCsv() {
        if (!dump.isSelected()){
            return;
        }
        if (csvPath == null) {
            LOGGER.error("csv_path is null.");
            return;
        }
        File file = new File(csvPath);

        if (file.isFile() && file.getName().endsWith(".csv")) {
            ShowResult showResult = new ShowResult(project, file.getPath());
            showResult.show();
        }
    }
}
