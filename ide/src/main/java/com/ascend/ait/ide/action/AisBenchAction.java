package com.ascend.ait.ide.action;

import com.ascend.ait.ide.Icons;
import com.ascend.ait.ide.commlib.ui.UiUtils;
import com.ascend.ait.ide.optimizie.ui.step.ais_bench_basic;
import com.intellij.openapi.actionSystem.AnAction;
import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.util.IconLoader;
import org.jetbrains.annotations.NotNull;

public class AisBenchAction extends AnAction {

    public AisBenchAction() {
        super("AisBench", "", UiUtils.getJbIcon(Icons.AIS_BENCH_DARK, Icons.AIS_BENCH_LIGHT));
    }

    @Override
    public void actionPerformed(@NotNull AnActionEvent e) {
        ais_bench_basic basic = new ais_bench_basic(e.getProject());
        basic.show();
    }
}
