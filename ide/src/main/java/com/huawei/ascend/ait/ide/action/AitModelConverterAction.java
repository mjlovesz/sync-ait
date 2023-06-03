package com.huawei.ascend.ait.ide.action;

import com.huawei.ascend.ait.ide.Icons;
import com.huawei.ascend.ait.ide.commonlib.ui.UiUtils;
import com.huawei.ascend.ait.ide.optimizie.ui.step.AitModelConverterStep;
import com.intellij.openapi.actionSystem.AnAction;
import com.intellij.openapi.actionSystem.AnActionEvent;
import org.jetbrains.annotations.NotNull;

public class AitModelConverterAction extends AnAction {

    public AitModelConverterAction() {
        super("AitModelConverter", "", UiUtils.getJbIcon(Icons.AIT_MODEL_CONVERTER_DARK, Icons.AIT_MODEL_CONVERTER_LIGHT));
    }

    @Override
    public void actionPerformed(AnActionEvent event) {
        if (event.getProject() == null) {
            return;
        }

        AitModelConverterStep aitModelConverterStep = new AitModelConverterStep(event.getProject());
        aitModelConverterStep.show();
    }

    @Override
    public void update(@NotNull AnActionEvent event) {
        super.update(event);
        if (event.getProject() == null) {
            return;
        }

        event.getPresentation().setEnabledAndVisible(true);
//        if (OS.isLinux()) {
//            event.getPresentation().setEnabledAndVisible(true);
//        } else {
//            event.getPresentation().setEnabledAndVisible(false);
//        }
    }
}
