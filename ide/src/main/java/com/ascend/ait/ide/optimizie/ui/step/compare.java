package com.ascend.ait.ide.optimizie.ui.step;

import com.ascend.ait.ide.commonlib.exception.CommandInjectException;
import com.ascend.ait.ide.commonlib.output.OutputService;
import com.ascend.ait.ide.commonlib.ui.SwitchButton;
import com.ascend.ait.ide.commonlib.util.safeCmd.CmdExec;
import com.ascend.ait.ide.commonlib.util.safeCmd.CmdStrBuffer;
import com.intellij.execution.ui.ConsoleViewContentType;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.ui.DialogWrapper;
import com.intellij.openapi.ui.TextFieldWithBrowseButton;
import org.jetbrains.annotations.Nullable;

import javax.swing.*;
import java.io.IOException;

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
