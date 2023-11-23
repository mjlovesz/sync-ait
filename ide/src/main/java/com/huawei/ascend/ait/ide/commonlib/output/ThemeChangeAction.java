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

package com.huawei.ascend.ait.ide.commonlib.output;

import com.huawei.ascend.ait.ide.commonlib.icons.CommonLibIcons;
import com.intellij.openapi.actionSystem.AnAction;
import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.wm.ToolWindow;
import com.intellij.openapi.wm.ToolWindowManager;

import org.jetbrains.annotations.NotNull;

import javax.swing.UIManager;

/**
 * ThemeChangeAction
 *
 * @author cabbage
 * @since 2023/06/03
 */
public class ThemeChangeAction extends AnAction {
    @Override
    public void actionPerformed(@NotNull AnActionEvent actionEvent) {
        // not need execute
    }

    @Override
    public void update(@NotNull AnActionEvent event) {
        super.update(event);
        event.getPresentation().setVisible(true);
        if (event.getProject() != null) {
            setToolWindowIcon(event);
        }
        event.getPresentation().setVisible(false);
    }

    /**
     * set toolwindows icon
     *
     * @param event action event
     */
    public void setToolWindowIcon(@NotNull AnActionEvent event) {
        ToolWindow outputToolWindows = ToolWindowManager.getInstance(event.getProject()).getToolWindow("Output");
        if (outputToolWindows != null) {
            if (UIManager.getLookAndFeel().getName().contains("Darcula")
                    || UIManager.getLookAndFeel().getName().contains("Dark")) {
                outputToolWindows.setIcon(CommonLibIcons.TOOL_ICON_DARK);
                OutputFactory.setConsoleContent(CommonLibIcons.TOOL_ICON_DARK);
                OutputFactory.setDetailsContent(CommonLibIcons.DETAIL_ICON_DARK);
            } else {
                outputToolWindows.setIcon(CommonLibIcons.TOOL_ICON);
                OutputFactory.setConsoleContent(CommonLibIcons.TOOL_ICON);
                OutputFactory.setDetailsContent(CommonLibIcons.DETAIL_ICON);
            }
        }
    }
}