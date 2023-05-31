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

package com.ascend.ait.ide.optimizie.ui.step;

import com.ascend.ait.ide.commonlib.ui.SwitchButton;
import com.ascend.ait.ide.commonlib.util.safeCmd.CmdStrBuffer;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.ui.DialogWrapper;
import com.intellij.openapi.ui.TextFieldWithBrowseButton;
import org.jetbrains.annotations.Nullable;

import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

public class compare extends DialogWrapper {
    private JPanel root;
    private JLabel model;
    private TextFieldWithBrowseButton modelFileBrowse;
    private JTextField loopTextField;
    private JLabel debug;
    private SwitchButton debugButton;
    private final Project project;

    public compare(@Nullable Project project) {
        super(true);
        this.project = project;
        init();
        setTitle("Ais Bench");
        setIcons();

        setFileChoodeAction();
        initComponent();
        setOKButtonText("Start");
    }

    private void setIcons() {
    }

    private void setFileChoodeAction() {
    }

    private void initComponent() {
    }

    @Override
    protected void doOKAction() {
        close(0);
    }

    private CmdStrBuffer getCmdStrBuffer() {
        return null;
    }

    private Boolean preCheck() {
        return true;
    }

    @Override
    protected @Nullable JComponent createCenterPanel() {
        return root;
    }
}
