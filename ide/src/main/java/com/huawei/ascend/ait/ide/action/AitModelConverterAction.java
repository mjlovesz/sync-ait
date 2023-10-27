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
import com.huawei.ascend.ait.ide.optimizie.ui.step.AitModelConverterStep;
import com.intellij.openapi.actionSystem.AnAction;
import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.project.Project;
import org.cef.OS;
import org.jetbrains.annotations.NotNull;

/**
 * AitModelConvertTask
 *
 * @author Jinhaiyang
 * @since 2023/06/03
 */
public class AitModelConverterAction extends AnAction {
    public AitModelConverterAction() {
        super("AitModelConverter", "",
                UiUtils.getJbIcon(Icons.MODEL_CONVERT_TITLE_DARK, Icons.MODEL_CONVERT_TITLE_LIGHT));
    }

    @Override
    public void actionPerformed(AnActionEvent event) {
        if (event.getProject() == null) {
            return;
        }

        openNewPage(event.getProject());
    }

    @Override
    public void update(@NotNull AnActionEvent event) {
        super.update(event);
        if (event.getProject() == null) {
            return;
        }

        event.getPresentation().setEnabledAndVisible(OS.isLinux());
    }

    /**
     * Open New Page
     *
     * @param project project
     */
    public void openNewPage(@NotNull Project project) {
        AitModelConverterStep aitModelConverterStep = new AitModelConverterStep(project);
        aitModelConverterStep.show();
    }
}
