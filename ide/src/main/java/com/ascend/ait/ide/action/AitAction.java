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
        super("AIT", "",
                UiUtils.getJbIcon(Icons.AIS_BENCH_DARK, Icons.AIS_BENCH_LIGHT));
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