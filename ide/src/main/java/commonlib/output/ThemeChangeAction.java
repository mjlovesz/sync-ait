

package com.ascend.ait.ide.commonlib.output;

import com.ascend.ait.ide.commonlib.icons.CommonLibIcons;
import com.intellij.openapi.actionSystem.AnAction;
import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.wm.ToolWindow;
import com.intellij.openapi.wm.ToolWindowManager;

import org.jetbrains.annotations.NotNull;

import javax.swing.UIManager;

/**
 * description
 *
 * @author Arther Yang
 * @since 2019/11/30
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
            if (UIManager.getLookAndFeel().getName().contains("Darcula") ||
                    UIManager.getLookAndFeel().getName().contains("Dark")) {
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