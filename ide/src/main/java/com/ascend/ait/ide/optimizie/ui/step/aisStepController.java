package com.ascend.ait.ide.optimizie.ui.step;

import com.ascend.ait.ide.util.BaseStepController;
import com.intellij.ide.util.projectWizard.ModuleWizardStep;
import com.intellij.ide.wizard.AbstractWizard;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.util.NlsContexts;
import org.jetbrains.annotations.NonNls;
import org.jetbrains.annotations.Nullable;

public class aisStepController extends BaseStepController {

    public aisStepController(String title, Project project) {
        super(title, project);
        initComponent();
    }

    @Override
    protected void initComponent() {

    }

    @Override
    protected @Nullable @NonNls String getHelpID() {
        return "";
    }
}
