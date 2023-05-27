package com.ascend.ait.ide.optimizie.ui;

import com.ascend.ait.ide.Icons;
import com.ascend.ait.ide.commonlib.pluginTool.PluginGet;
import com.ascend.ait.ide.commonlib.pluginTool.PluginClassId;
import com.ascend.ait.ide.commonlib.output.OutputFactory;
import com.ascend.ait.ide.commonlib.output.OutputService;
import com.ascend.ait.ide.commonlib.ui.UiUtils;
import com.intellij.ide.plugins.PluginManager;
import com.intellij.openapi.extensions.PluginId;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.wm.WindowManager;
import com.intellij.ui.JBColor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.HashMap;
import java.util.Optional;
import java.util.function.Function;

public class Choosedialog extends JFrame {
    private JPanel root;
    private JPanel guide;
    private JPanel leftJPanel;
    private JLabel rightIcon;
    private JLabel modelAnalyse;
    private JLabel aisBench;
    private JLabel compare;
    private JLabel systemProfiler;
    private JLabel modelConverter;
    private JLabel title;
    private JLabel step1;
    private JLabel step2;
    private JLabel step3;
    private JLabel step4;
    private JLabel step5;
    private JPanel modelAnalyseJpanel;
    private JPanel modelConverterJPanel;
    private JPanel aisBenchJPanel;
    private JPanel compareJPanel;
    private JPanel systemProfilerJPanel;
    private JPanel rightJPanel;
    private JPanel stepJPanel;

    private final Project project;
    private static final Logger LOG = LoggerFactory.getLogger(Choosedialog.class);
    private static final Color GUIDE_VIEW_COLOR = new JBColor(0xFFFFFF, 0x252425);
    private static final Color JLABEL_FONT_COLOR = new Color(0x007AFF);
    private static final Color HIGHT_LIGHT_COLOR = new Color(0xED9121);
    private static final Color TITLE_FONT_COLOR = new JBColor(0x777777, 0xCBCBCB);
    private static final String COMMON_FONT_STYLE = "Microsoft YaHei UI";
    private final HashMap<StepNum, Function<Project, ?>> actionMappings = new HashMap<>();
    private static final int ICON_TEXT_GAP = 12;
    private static final int JLABEL_FONT_SIZE = 20;
    private static final int TITLE_FONT_SIZE = 36;

    public Choosedialog(Project project) {
        this.project = project;
        initComponent();
        initIcon();
        setBackgroundColor();
        ModelAnalyse modelAnalyse = new ModelAnalyse();
        ModelConvert modelConvert = new ModelConvert();
        SystemProfiling systemProfiling = new SystemProfiling();
        AisBench aisBench = new AisBench();
        Compare compare = new Compare();
    }

    private void initIcon() {
        rightIcon.setIcon(UiUtils.getJbIcon(Icons.RIGHT_DARK, Icons.RIGHT_LIGHT));
        title.setFont(new Font(COMMON_FONT_STYLE, Font.PLAIN, TITLE_FONT_SIZE));
        title.setForeground(TITLE_FONT_COLOR);
    }

    private void setBackgroundColor() {
        root.setBackground(GUIDE_VIEW_COLOR);
        guide.setBackground(GUIDE_VIEW_COLOR);
        leftJPanel.setBackground(GUIDE_VIEW_COLOR);
        rightJPanel.setBackground(GUIDE_VIEW_COLOR);
        stepJPanel.setBackground(GUIDE_VIEW_COLOR);
    }

    private void activeOutput() {
        Window window = WindowManager.getInstance().suggestParentWindow(project);
        if (window == null) {
            LOG.warn("window is null");
            return;
        }
        OutputService.getInstance(project).active();
        OutputFactory.show(project);
    }

    private void setIcon(JLabel jLabel, Icon dark, Icon light) {
        jLabel.setIcon(UiUtils.getJbIcon(dark, light));
        jLabel.setIconTextGap(ICON_TEXT_GAP);
    }

    private void initComponent() {
        root.setVisible(true);
        initMouse();
        root.addPropertyChangeListener(proprtyChangeEvent -> {
           initIcon();
        });
    }

    public JComponent getRoot() {
        return root;
    }

    private void checkPluginsAndAddAction(JLabel jLabel, StepNum step, String  pluginClass) {
        if (PluginId.findId(pluginClass) != null && PluginManager.getInstance().findEnabledPlugin(PluginId.getId(pluginClass)) != null) {
            jLabel.addMouseListener(new ChooseViewMouseAdapter(step, jLabel));
        }
    }

    private void initMouse() {
        checkPluginsAndAddAction(modelAnalyse, StepNum.MODEL_ANALYSE, PluginClassId.ModelAnalyse_ClassId);
        checkPluginsAndAddAction(modelConverter, StepNum.MODEL_CONVERTER, PluginClassId.ModelConverter_ClassId);
        checkPluginsAndAddAction(aisBench, StepNum.AIS_BENCH, "test");
        checkPluginsAndAddAction(compare, StepNum.COMPARE, "ls");
        checkPluginsAndAddAction(systemProfiler, StepNum.SYSTEM_PROFILER, "test");
    }

    private class ModelAnalyse {
        ModelAnalyse() {
            setIcon(modelAnalyse, Icons.MODEL_ANALYSE_DARK, Icons.MODEL_ANALYSE_LIGHT);
            setStepIcons(modelAnalyseJpanel, modelAnalyse, step1, PluginClassId.ModelAnalyse_ClassId);
            actionMappings.put(StepNum.MODEL_ANALYSE, this::doModelAnalyse);
        }

        private Object doModelAnalyse(Project project){
            activeOutput();
            PluginGet pluginGet = new PluginGet(project);
            pluginGet.getPluginClass(PluginClassId.ModelAnalyse_ClassId, PluginClassId.Inference_PluginId);
            return Optional.empty();
        }
    }

    private class ModelConvert {
        ModelConvert() {
            setIcon(modelConverter, Icons.MODEL_CONVERTER_DARK, Icons.MODEL_CONVERTER_LIGHT);
            setStepIcons(modelConverterJPanel, modelConverter, step2, PluginClassId.ModelConverter_ClassId);
            actionMappings.put(StepNum.MODEL_CONVERTER, this::doModelConvert);
        }

        private Object doModelConvert(Project project){
            activeOutput();
            PluginGet pluginGet = new PluginGet(project);
            pluginGet.getPluginClass(PluginClassId.ModelConverter_ClassId, PluginClassId.Inference_PluginId);
            return Optional.empty();
        }
    }
    private class AisBench {
        AisBench() {
            setIcon(aisBench, Icons.AIS_BENCH_DARK, Icons.AIS_BENCH_LIGHT);
            setStepIcons(aisBenchJPanel, aisBench, step3, PluginClassId.AisBench_ClassId);
            actionMappings.put(StepNum.AIS_BENCH, this::doAisBench);
        }

        private Object doAisBench(Project project){
            activeOutput();
            PluginGet pluginGet = new PluginGet(project);
            pluginGet.getPluginClass(PluginClassId.AisBench_ClassId, PluginClassId.AitIde_PluginId);
            return Optional.empty();
        }
    }

    private class Compare {
        Compare() {
            setIcon(compare, Icons.COMPARE_DARK, Icons.COMPARE_LIGHT);
            setStepIcons(compareJPanel, compare, step4, PluginClassId.ModelConverter_ClassId);
            actionMappings.put(StepNum.COMPARE, this::doCompare);
        }

        private Object doCompare(Project project){
            activeOutput();
            PluginGet pluginGet = new PluginGet(project);
            pluginGet.getPluginClass(PluginClassId.ModelConverter_ClassId, PluginClassId.AitIde_PluginId);
            return Optional.empty();
        }
    }
    private class SystemProfiling {
        SystemProfiling() {
            setIcon(systemProfiler, Icons.SYSTEM_PROFILER_DARK, Icons.SYSTEM_PROFILER_LIGHT);
            setStepIcons(systemProfilerJPanel, systemProfiler, step5, PluginClassId.SystemProfiler_ClassId);
            actionMappings.put(StepNum.MODEL_CONVERTER, this::doSystemProfiling);
        }

        private Object doSystemProfiling(Project project){
            activeOutput();
            PluginGet pluginGet = new PluginGet(project);
            pluginGet.getPluginClass(PluginClassId.SystemProfiler_ClassId, PluginClassId.Foundation_PluginId);
            return Optional.empty();
        }
    }

    public void setStepIcons(JPanel jPanel, JLabel jLabel1, JLabel jLabel2, String  pluginClass) {
        jPanel.setBackground(GUIDE_VIEW_COLOR);

        jLabel1.setBackground(GUIDE_VIEW_COLOR);
        jLabel1.setFont(new Font(COMMON_FONT_STYLE, Font.PLAIN, JLABEL_FONT_SIZE));
        if (PluginId.findId(pluginClass) != null && PluginManager.getInstance().findEnabledPlugin(PluginId.getId(pluginClass)) != null) {
            jLabel1.setForeground(JLABEL_FONT_COLOR);
        } else {
            jLabel1.setForeground(TITLE_FONT_COLOR);
        }

        jLabel2.setFont(new Font(COMMON_FONT_STYLE, Font.PLAIN, JLABEL_FONT_SIZE));
        jLabel2.setForeground(TITLE_FONT_COLOR);
    }

    private class ChooseViewMouseAdapter extends MouseAdapter {
        private final JLabel jLabel;
        private final StepNum action;
        private String jLabelText;

        private ChooseViewMouseAdapter(StepNum stepNum, JLabel jLabel) {
            this.jLabel = jLabel;
            this.action = stepNum;
        }

        @Override
        public void mouseClicked(MouseEvent e) {
            jLabel.setForeground(HIGHT_LIGHT_COLOR);
            actionMappings.get(action).apply(project);
        }

        @Override
        public void mouseEntered(MouseEvent e) {
            jLabelText = jLabel.getText();
            jLabel.setText("<HTML><U>" + jLabelText + "</U></HTML>");
        }

        @Override
        public void mouseExited(MouseEvent e) {
            jLabel.setForeground(JLABEL_FONT_COLOR);
            jLabel.setText(jLabelText);
        }
    }

    public enum StepNum {
        MODEL_ANALYSE,
        MODEL_CONVERTER,
        AIS_BENCH,
        COMPARE,
        SYSTEM_PROFILER

    }


}
