/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

package com.huawei.mindstudio.output;

import com.huawei.mindstudio.output.icons.OutputIcons;
import com.intellij.execution.filters.TextConsoleBuilderFactory;
import com.intellij.execution.impl.ConsoleViewImpl;
import com.intellij.execution.ui.ConsoleView;
import com.intellij.openapi.actionSystem.*;
import com.intellij.openapi.application.ApplicationManager;
import com.intellij.openapi.editor.Editor;
import com.intellij.openapi.editor.actions.ToggleUseSoftWrapsInPreviewAction;
import com.intellij.openapi.editor.actions.ToggleUseSoftWrapsToolbarAction;
import com.intellij.openapi.editor.impl.softwrap.SoftWrapAppliancePlaces;
import com.intellij.openapi.project.DumbAware;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.wm.ToolWindow;
import com.intellij.openapi.wm.ToolWindowAnchor;
import com.intellij.openapi.wm.ToolWindowFactory;
import com.intellij.openapi.wm.ToolWindowManager;
import com.intellij.ui.content.Content;
import com.intellij.ui.content.ContentManager;
import kotlinx.html.A;
import org.jetbrains.annotations.NotNull;

import javax.swing.JPanel;
import javax.swing.JComponent;
import javax.swing.Icon;
import javax.swing.UIManager;
import java.awt.BorderLayout;
import java.util.HashMap;
import java.util.Map;

/**
 * OutputFactory
 *
 * @author: zhouxiaohui
 * @since: 2019/06/05 10:10
 * @version: 1.0
 */
public class OutputFactory implements ToolWindowFactory, DumbAware{
    private static final String OUTPUT_TOOL_WINDOW_ID = "Output";

    private static final String OUTPUT_TOOL_WINDOW_DETAILS = "Detail";

    private static final String DARCULA = "Darcula";

    private static final String DARK = "Dark";

    private static ConsoleView console;

    private static ConsoleView details;

    private static Map<Project, ConsoleView> normalConsoleViewmap = new HashMap<>();

    private static Map<Project, ConsoleView> detailConsoleViewmap = new HashMap<>();

    private static Content consoleContent;
    private static Content detailsContent;

    public static Map<Project, ConsoleView> getNormalConsoleViewmap() {
        return normalConsoleViewmap;
    }

    public static Map<Project, ConsoleView> getDetailConsoleViewmap() {
        return detailConsoleViewmap;
    }

    private static void setContentTheme(Content content) {
        if (UIManager.getLookAndFeel().getName().contains(DARCULA) || UIManager.getLookAndFeel().getName().contains(DARK)) {
            content.setIcon(OutputIcons.DETAIL_ICON_DARK);
        } else {
            content.setIcon(OutputIcons.DETAIL_ICON);
        }
    }

    private static void setToolWindowTheme(ToolWindow toolWindow) {
        if (UIManager.getLookAndFeel().getName().contains(DARCULA) || UIManager.getLookAndFeel().getName().contains(DARK)) {
            toolWindow.setIcon(OutputIcons.TOOL_ICON_DARK);
        } else {
            toolWindow.setIcon(OutputIcons.TOOL_ICON);
        }
    }

    private static AnAction getConsoleAction(AnAction consoleAction) {
        if (!(consoleAction instanceof ToggleUseSoftWrapsToolbarAction && console instanceof ConsoleViewImpl)) {
            return consoleAction;
        }
        ConsoleViewImpl consoleView = (ConsoleViewImpl) console;
        return new ToggleUseSoftWrapsToolbarAction(SoftWrapAppliancePlaces.CONSOLE) {
            private boolean isSelected = false;
            @Override
            protected Editor getEditor(@NotNull AnActionEvent ex) {return consoleView.getEditor();}
            @Override
            public void update(@NotNull AnActionEvent ex) {
                super.update(ex);
                boolean isNewSelect = isSelected(ex);
                if (isNewSelect != isSelected) {
                    isSelected = isNewSelect;
                    ApplicationManager.getApplication().invokeLater(() -> setSelected(ex, isNewSelect));
                }
            }
        };
    }

    /**
     * active output toolwindow
     *
     * @param project current project
     */
    public static void activate(Project project) {
        ToolWindow toolWindow = ToolWindowManager.getInstance(project).getToolWindow(OUTPUT_TOOL_WINDOW_ID);
        if (toolWindow == null){
            return;
        }
        ApplicationManager.getApplication().invokeLater(() -> {
            if (!project.isDisposed()) {
                toolWindow.setAvailable(true);
                toolWindow.activate(null);
            }
        });
    }

    /**
     * sync show output toolwindow
     *
     * @param project current project
     */
    public static void show(Project project) {


        ToolWindow toolWindow = ToolWindowManager.getInstance(project).getToolWindow(OUTPUT_TOOL_WINDOW_ID);
        if (toolWindow == null){
            return;
        }
        ApplicationManager.getApplication().invokeAndWait(() ->
                {
                    if (!project.isDisposed()) {
                        toolWindow.show(null);
                    }
                }
        );
    }

    /**
     * set icon for output of output toolwindow
     *
     * @param TOOL_ICON tool icon
     */
    public static void setConsoleContent(Icon TOOL_ICON) {
        if (consoleContent != null) {
            consoleContent.setIcon(TOOL_ICON);
        }
    }

    /**
     * set icon for detail of output toolwindow
     *
     * @param TOOL_ICON tool icon
     */
    public static void setDetailsContent(Icon TOOL_ICON) {
        if (detailsContent != null) {
            detailsContent.setIcon(TOOL_ICON);
        }
    }

    @Override
    public void createToolWindowContent(@NotNull Project project, @NotNull ToolWindow toolWindow) {
        if (project.isDisposed()) {
            return;
        }
        console = TextConsoleBuilderFactory.getInstance().createBuilder(project).getConsole();
        JComponent component = console.getComponent();
        DefaultActionGroup consoleActions = new DefaultActionGroup();
        for (AnAction anAction : console.createConsoleActions()) {
            consoleActions.add(getConsoleAction(consoleActions));
        }

        ActionToolbar actionToolbar = ActionManager.getInstance().createActionToolbar(ActionPlaces.TOOLBAR,
                consoleActions, false);
        actionToolbar.setTargetComponent(component);

        JPanel consolePanel = new JPanel(new BorderLayout());
        consolePanel.add(actionToolbar.getComponent(), BorderLayout.WEST);

    }

    @Override
    public void init(@NotNull ToolWindow toolWindow) {
        ToolWindowFactory.super.init(toolWindow);
        toolWindow.setTitle(OUTPUT_TOOL_WINDOW_ID);
        setToolWindowTheme(toolWindow);
    }

    @Override
    public boolean shouldBeAvailable(@NotNull Project project) {
        return ToolWindowFactory.super.shouldBeAvailable(project);
    }
}
