package com.ascend.ait.ide.action;

import com.ascend.ait.ide.Icons;
import com.ascend.ait.ide.commonlib.ui.UiUtils;
import com.ascend.ait.ide.optimizie.ui.step.ais_bench_basic;
import com.ascend.ait.ide.optimizie.ui.step.compare;
import com.intellij.openapi.actionSystem.AnAction;
import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.actionSystem.Presentation;
import com.intellij.openapi.fileEditor.FileEditorManager;
import com.intellij.openapi.project.Project;
import com.intellij.testFramework.LightVirtualFile;
import org.jetbrains.annotations.NotNull;

public class CompareAction extends AnAction {

    public CompareAction() {
        super("Compare", "", UiUtils.getJbIcon(Icons.COMPARE_DARK, Icons.COMPARE_LIGHT));
    }
    @Override
    public void actionPerformed(@NotNull AnActionEvent e) {
        if (e.getProject() == null) {
            return;
        }
        openNewPage(e.getProject());
    }

    public void openNewPage(@NotNull Project project) {
        compare c = new compare(project);
        c.show();
    }
}