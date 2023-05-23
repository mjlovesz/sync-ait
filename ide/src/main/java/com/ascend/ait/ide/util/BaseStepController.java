package com.ascend.ait.ide.util;

import com.intellij.ide.util.projectWizard.ModuleWizardStep;
import com.intellij.ide.wizard.AbstractWizard;
import com.intellij.openapi.components.ServiceManager;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.util.NlsContext;
import com.intellij.openapi.util.NlsContexts;
import org.jetbrains.annotations.NonNls;
import org.jetbrains.annotations.Nullable;

import java.awt.*;
import java.lang.reflect.Field;

public abstract class BaseStepController extends AbstractWizard<ModuleWizardStep> {

    private final Project project;

    public BaseStepController(String title, Project project) {
        super(title, project);
        this.project = project;
        this.getWindow().setMinimumSize(new Dimension(586, 210));
    }

    protected abstract void initComponent();

    protected void clearAbstractWizardButton(String buttonName) {
        try {
            Field  buttonField = AbstractWizard.class.getDeclaredField(buttonName);
            buttonField.setAccessible(true);
            Object button = buttonField.get(this);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void doNextAction() {
        super.doNextAction();
        setStartName();
    }

    private void setStartName() {
        getNextButton().setText("start");
    }

    @Override
    public void dispose() {
        super.dispose();
    }


    @Override
    protected @Nullable @NonNls String getHelpID() {
        return null;
    }
}
