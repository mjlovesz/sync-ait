package com.ascend.ait.ide.optimizie.ui.step;

import com.intellij.openapi.project.Project;
import com.intellij.openapi.ui.DialogWrapper;
import org.jetbrains.annotations.Nullable;

import javax.swing.*;

public class compare extends DialogWrapper {
    private JPanel root;

    public compare(@Nullable Project project) {
        super(project);
    }

    @Override
    protected @Nullable JComponent createCenterPanel() {
        return root;
    }
}
