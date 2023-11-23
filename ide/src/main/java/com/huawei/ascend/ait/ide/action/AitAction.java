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

package com.huawei.ascend.ait.ide.action;

import com.huawei.ascend.ait.ide.Icons;
import com.huawei.ascend.ait.ide.commonlib.ui.UiUtils;

import com.intellij.openapi.actionSystem.AnAction;
import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.fileEditor.FileEditorManager;
import com.intellij.openapi.project.Project;
import com.intellij.testFramework.LightVirtualFile;

/**
 * AitAction
 *
 * @author cabbage
 * @since 2023/06/03
 */
public class AitAction extends AnAction {
    /**
     * AitAction
     */
    public AitAction() {
        super("Ait", "", UiUtils.getJbIcon(Icons.AIT_TITLE_DARK, Icons.AIT_TITLE_LIGHT));
    }

    @Override
    public void actionPerformed(AnActionEvent e) {
        if (e.getProject() == null) {
            return;
        }
        openEditor(e.getProject());
    }

    /**
     * open Editor
     *
     * @param project project
     */
    private void openEditor(Project project) {
        FileEditorManager instance = FileEditorManager.getInstance(project);
        instance.openFile(new LightVirtualFile("AIT"), true);
    }
}