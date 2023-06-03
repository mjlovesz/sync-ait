package com.huawei.ascend.ait.ide.optimizie.aitmodelconvert;

import com.huawei.ascend.ait.ide.commonlib.exception.CommandInjectException;
import com.huawei.ascend.ait.ide.commonlib.output.OutputService;
import com.huawei.ascend.ait.ide.commonlib.util.safeCmd.CmdExec;
import com.huawei.ascend.ait.ide.commonlib.util.safeCmd.CmdStrBuffer;
import com.huawei.ascend.ait.ide.commonlib.util.safeCmd.CmdStrWordStatic;
import com.intellij.execution.ui.ConsoleViewContentType;
import com.intellij.openapi.application.ApplicationManager;
import com.intellij.openapi.progress.ProgressIndicator;
import com.intellij.openapi.progress.Task.Backgroundable;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.project.ProjectManager;
import com.intellij.openapi.project.ProjectManagerListener;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;

public class AitModelConvertTask extends Backgroundable {
    private final Project project;
    private final String modelFile;
    private final String outputModelFile;
    private final String socVersion;
    private CmdExec exec;
    private final Object lock = new Object();
    private boolean isTaskFinished;

    public AitModelConvertTask(@NotNull Project project, String modelFile, String socVersion, String outputModelFile) {
        super(project, "Converting model...", true);
        this.project = project;
        this.modelFile = modelFile;
        this.socVersion = socVersion;
        this.outputModelFile = outputModelFile;
        ProjectManager.getInstance().addProjectManagerListener(project, new ProjectManagerListener() {
            @Override
            public void projectClosing(@NotNull Project project) {
                if (!isTaskFinished) {
                    stopConvertModel();
                }
            }
        });
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
                }
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
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
        cmd.append("ait").append(CmdStrWordStatic.SPACE)
                .append("convert").append(CmdStrWordStatic.SPACE)
                .append("--model").append(CmdStrWordStatic.SPACE)
                .append(modelFile).append(CmdStrWordStatic.SPACE)
                .append("--output").append(CmdStrWordStatic.SPACE)
                .append(outputModelFile).append(CmdStrWordStatic.SPACE)
                .append("--soc_version").append(CmdStrWordStatic.SPACE)
                .append(socVersion);

        return cmd;
    }

    private boolean doModelConvert(@NotNull ProgressIndicator indicator) {


        synchronized (lock) {
            exec = new CmdExec();
        }

        boolean convertSuccess = false;
        OutputService.getInstance(project).print("Start converting the model.", ConsoleViewContentType.LOG_INFO_OUTPUT);
        OutputService.getInstance(project).print("Convert model environment variables:", ConsoleViewContentType.LOG_INFO_OUTPUT);
        OutputService.getInstance(project).print("xxx");
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
