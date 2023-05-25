package com.ascend.ait.ide.optimizie.ui;

import com.ascend.ait.ide.Icons;
import com.ascend.ait.ide.optimizie.ui.step.ais_bench_basic;
import com.ascend.ait.ide.commonlib.output.OutputFactory;
import com.ascend.ait.ide.commonlib.output.OutputService;
import com.ascend.ait.ide.commonlib.ui.UiUtils;
import com.intellij.openapi.project.Project;

import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

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
        rightIcon.setIcon(UiUtils.getJbIcon(Icons.RIGHT_DARK, Icons.RIGHT_LIGHT));
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
