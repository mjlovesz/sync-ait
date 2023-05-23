package com.ascend.ait.ide.optimizie.ui;

import com.ascend.ait.ide.optimizie.ui.step.aisStepController;
import com.ascend.ait.ide.optimizie.ui.step.ais_bench_basic;
import com.huawei.mindstudio.output.OutputFactory;
import com.huawei.mindstudio.output.OutputService;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.util.IconLoader;

import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.io.IOException;

public class Choosedialog extends JFrame {
    private JPanel root;
    private JPanel guide;
    private JPanel left;
    private JLabel rightIcon;
    private JLabel ModelTest;

    private final Project project;

    public Choosedialog(Project project) {
        this.project = project;
        initComponent();
        initIcon();
        setAction();
    }

    private void initIcon() {
        rightIcon.setIcon(IconLoader.findIcon("/icons/dark@1x.png", Choosedialog.class));
    }

    private void initComponent() {
        root.setVisible(true);
    }

    public JComponent getRoot() {
        return root;
    }

    private void setAction() {
        ModelTest.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                OutputService.getInstance(project).active();
                OutputFactory.show(project);
                ais_bench_basic basic = new ais_bench_basic(project);
                basic.show();
            }

            @Override
            public void mouseEntered(MouseEvent e) {
                super.mouseEntered(e);
            }

            @Override
            public void mouseExited(MouseEvent e) {
                super.mouseExited(e);
            }
        });
    }

}
