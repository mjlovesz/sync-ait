package com.ascend.ait.ide.action;

import com.ascend.ait.ide.Icons;
import com.ascend.ait.ide.commonlib.ui.UiUtils;
import com.intellij.openapi.actionSystem.AnAction;
import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.actionSystem.Presentation;
import com.intellij.openapi.fileEditor.FileEditorManager;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.util.IconLoader;
import com.intellij.testFramework.LightVirtualFile;
import org.jetbrains.annotations.NotNull;

public class AitAction extends AnAction {

    public AitAction() {
        super("AIT", "", UiUtils.getJbIcon(Icons.AIS_BENCH_DARK, Icons.AIS_BENCH_LIGHT));
    }
    @Override
    public void actionPerformed(AnActionEvent e) {
        if (e.getProject() == null) {
            return;
        }
        openEditor(e.getProject());
    }

    private void openEditor(Project project) {
        FileEditorManager instance = FileEditorManager.getInstance(project);
        instance.openFile(new LightVirtualFile("AIT"), true);
    }

    @Override
    public void update(@NotNull AnActionEvent e) {
        super.update(e);
        Presentation presentation = e.getPresentation();
        Project project = e.getProject();
        if (project == null) {
            presentation.setVisible(false);
            return;
        }
        presentation.setVisible(true);
        presentation.setIcon(UiUtils.getJbIcon(Icons.AIS_BENCH_DARK, Icons.AIS_BENCH_LIGHT));
    }

}