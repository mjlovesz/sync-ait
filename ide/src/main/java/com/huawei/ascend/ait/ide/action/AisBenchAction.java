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
import com.huawei.ascend.ait.ide.commonlib.output.OutputService;
import com.huawei.ascend.ait.ide.commonlib.ui.UiUtils;
import com.huawei.ascend.ait.ide.optimizie.ui.step.AisBenchBasic;

import com.intellij.openapi.actionSystem.AnAction;
import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.project.Project;

import com.intellij.openapi.wm.WindowManager;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.Window;

/**
 * Ais bench
 *
 * @author cabbage
 * @since 2023/06/03
 */
public class AisBenchAction extends AnAction {
    private static final Logger LOG = LoggerFactory.getLogger(AisBenchAction.class);

    /**
     * Ais bench
     */
    public AisBenchAction() {
        super("AisBench", "", UiUtils.getJbIcon(Icons.AIS_TITLE_DARK, Icons.AIS_TITLE_LIGHT));
    }

    @Override
    public void actionPerformed(@NotNull AnActionEvent e) {
        if (e.getProject() == null) {
            return;
        }
        openNewPage(e.getProject());
    }

    /**
     * Open New Page
     *
     * @param project project
     */
    public void openNewPage(@NotNull Project project) {
        Window window = WindowManager.getInstance().suggestParentWindow(project);
        if (window == null) {
            LOG.warn("window is null");
            return;
        }
        OutputService.getInstance(project).active();
        AisBenchBasic basic = new AisBenchBasic(project);
        basic.show();
    }
}
