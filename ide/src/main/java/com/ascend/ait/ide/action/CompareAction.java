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
import com.ascend.ait.ide.optimizie.ui.step.Compare;
import com.intellij.openapi.actionSystem.AnAction;
import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.project.Project;
import org.jetbrains.annotations.NotNull;

public class CompareAction extends AnAction {

    public CompareAction() {
        super("Compare", "",
                UiUtils.getJbIcon(Icons.COMPARE_DARK, Icons.COMPARE_LIGHT));
    }
    @Override
    public void actionPerformed(@NotNull AnActionEvent e) {
        if (e.getProject() == null) {
            return;
        }
        openNewPage(e.getProject());
    }

    public void openNewPage(@NotNull Project project) {
        Compare compare = new Compare(project);
        compare.show();
    }
}