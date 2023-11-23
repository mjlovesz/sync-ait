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
import com.huawei.ascend.ait.ide.commonlib.util.safecmd.CmdExec;
import com.huawei.ascend.ait.ide.commonlib.util.safecmd.CmdStrBuffer;
import com.huawei.ascend.ait.ide.commonlib.util.safecmd.CmdStrWordStatic;
import com.intellij.execution.ui.ConsoleViewContentType;
import com.intellij.openapi.application.ApplicationManager;
import com.intellij.openapi.progress.ProgressIndicator;
import com.intellij.openapi.progress.Task.Backgroundable;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.project.ProjectManager;
import com.intellij.openapi.project.ProjectManagerListener;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.jetbrains.annotations.NotNull;

import java.io.File;
import java.io.IOException;

/**
 * AitModelConvertTask
 *
 * @author Jinhaiyang
 * @since 2023/06/03
 */
public class AitModelConvertTask extends Backgroundable {
    private final Project project;
    private final String cannPath;
    private final String aiePath;
    private final String modelFile;
    private final String outputModelFile;
    private final String socVersion;
    private CmdExec exec;
    private final Object lock = new Object();
    private final CmdStrWordStatic CMD_CONNECTOIN = new CmdStrWordStatic("&&");
    private final CmdStrWordStatic CMD_SOURCE = new CmdStrWordStatic("source");
    private final String SET_ENV_SH = "set_env.sh";

    public AitModelConvertTask(@NotNull Project project, String cannPath, String aiePath,
                               String modelFile, String socVersion, String outputModelFile) {
        super(project, "Converting model...", true);
        this.project = project;
        this.cannPath = cannPath;
        this.aiePath = aiePath;
        this.modelFile = modelFile;
        this.socVersion = socVersion;
        this.outputModelFile = outputModelFile;
        ProjectManager.getInstance().addProjectManagerListener(project, new ProjectManagerListener() {
            @Override
            public void projectClosing(@NotNull Project project) {
                stopConvertModel();
            }
        });
        OutputService.getInstance(project).clear();
        OutputService.getInstance(project).active();
        OutputService.getInstance(project).show();
    }

    public void stopConvertModel() {
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
                Thread.sleep(20);
                if (indicator.isCanceled()) {
                    OutputService.getInstance(project).print("Model convert has been stopped by the user.",
                                                             ConsoleViewContentType.LOG_ERROR_OUTPUT);
                    stopConvertModel();
                    return;
                }
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

    private CmdStrBuffer getModelConvertCommand() {
        CmdStrBuffer cmd = new CmdStrBuffer();
        String setEnvFilePath = StringUtils.join(String.valueOf(FileUtils.getFile(cannPath).toPath().getParent()), File.separator, SET_ENV_SH);

        cmd.append("export").append(CmdStrWordStatic.SPACE)
                .append("CANN_HOME=").append(cannPath).append(CmdStrWordStatic.SPACE)
                .append(CMD_CONNECTOIN).append(CmdStrWordStatic.SPACE)
                .append("export").append(CmdStrWordStatic.SPACE)
                .append("AIE_DIR=").append(aiePath).append(CmdStrWordStatic.SPACE)
                .append(CMD_CONNECTOIN).append(CmdStrWordStatic.SPACE)
                .append(CMD_SOURCE).append(CmdStrWordStatic.SPACE)
                .append(setEnvFilePath).append(CmdStrWordStatic.SPACE)
                .append(CMD_CONNECTOIN).append(CmdStrWordStatic.SPACE)
                .append("ait").append(CmdStrWordStatic.SPACE)
                .append("convert").append(CmdStrWordStatic.SPACE)
                .append("aie").append(CmdStrWordStatic.SPACE)
                .append("--golden-model").append(CmdStrWordStatic.SPACE)
                .append(modelFile).append(CmdStrWordStatic.SPACE)
                .append("--output-file").append(CmdStrWordStatic.SPACE)
                .append(outputModelFile).append(CmdStrWordStatic.SPACE)
                .append("--soc-version").append(CmdStrWordStatic.SPACE)
                .append(socVersion);

        return cmd;
    }

    private boolean doModelConvert(@NotNull ProgressIndicator indicator) {
        synchronized (lock) {
            exec = new CmdExec();
        }

        boolean convertSuccess = false;
        OutputService.getInstance(project).print("Start converting the model.", ConsoleViewContentType.LOG_INFO_OUTPUT);
        OutputService.getInstance(project).print("Convert model command:", ConsoleViewContentType.LOG_INFO_OUTPUT);
        CmdStrBuffer modelConvertCmd = getModelConvertCommand();
        OutputService.getInstance(project).print(modelConvertCmd.toString());
        try {
            exec.setStopCallback(indicator::isCanceled);
            if (exec.bashStart(modelConvertCmd)) {
                convertSuccess = true;
            }
        } catch (CommandInjectException | IOException exception) {
            OutputService.getInstance(project).print(exception.getMessage(), ConsoleViewContentType.LOG_INFO_OUTPUT);
        }

        if (!convertSuccess && (exec.getErrorResult() != null)) {
            OutputService.getInstance(project).print(exec.getErrorResult(), ConsoleViewContentType.LOG_INFO_OUTPUT);
        }

        synchronized (lock) {
            exec = null;
        }
        return convertSuccess;
    }

    private void convertModelSuccess() {
        OutputService.getInstance(project).print("Model converted successfully.", ConsoleViewContentType.LOG_INFO_OUTPUT);
        OutputService.getInstance(project).print("Model input path:" + modelFile);
        OutputService.getInstance(project).print("Model output path:" + outputModelFile);
        /* print log file */
    }

    private void convertModelFail() {
        OutputService.getInstance(project).print("Model conversion failure.", ConsoleViewContentType.LOG_ERROR_OUTPUT);
        OutputService.getInstance(project).print("Model input path:" + modelFile);
        /* print log file */
    }

    @Override
    public void run(@NotNull ProgressIndicator indicator) {
        OutputService.getInstance(project).scrollToEnd();
        executeThread(indicator);

        boolean convertSuccess = doModelConvert(indicator);
        if (convertSuccess) {
            convertModelSuccess();
        } else {
            convertModelFail();
        }
    }
}
