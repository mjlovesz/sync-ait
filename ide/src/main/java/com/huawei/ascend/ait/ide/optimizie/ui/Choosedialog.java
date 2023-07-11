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

package com.huawei.ascend.ait.ide.optimizie.ui;

import com.huawei.ascend.ait.ide.Icons;
import com.huawei.ascend.ait.ide.commonlib.pluginTool.PluginGet;
import com.huawei.ascend.ait.ide.commonlib.pluginTool.PluginClassId;
import com.huawei.ascend.ait.ide.commonlib.output.OutputFactory;
import com.huawei.ascend.ait.ide.commonlib.output.OutputService;
import com.huawei.ascend.ait.ide.commonlib.ui.UiUtils;

import com.intellij.ide.plugins.PluginManager;
import com.intellij.openapi.extensions.PluginId;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.wm.WindowManager;
import com.intellij.ui.JBColor;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.Icon;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import java.awt.Color;
import java.awt.Font;
import java.awt.Window;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.HashMap;
import java.util.Optional;
import java.util.function.Function;

/**
 * choosedialog
 *
 * @author cabbage
 * @date 2023/06/03
 */
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

    /**
     * choosedialog
     *
     * @param project project
     */
    public Choosedialog(Project project) {
        this.project = project;
        initComponent();
        initIcon();
        setBackgroundColor();
    }

    /**
     * init图标
     */
    private void initIcon() {
        rightIcon.setIcon(UiUtils.getJbIcon(Icons.RIGHT_DARK, Icons.RIGHT_LIGHT));
        title.setFont(new Font(COMMON_FONT_STYLE, Font.PLAIN, TITLE_FONT_SIZE));
        title.setForeground(TITLE_FONT_COLOR);
    }

    /**
     * 设置背景颜色
     */
    private void setBackgroundColor() {
        root.setBackground(GUIDE_VIEW_COLOR);
        guide.setBackground(GUIDE_VIEW_COLOR);
        leftJPanel.setBackground(GUIDE_VIEW_COLOR);
        rightJPanel.setBackground(GUIDE_VIEW_COLOR);
        stepJPanel.setBackground(GUIDE_VIEW_COLOR);
    }

    /**
     * activeOutput
     */
    private void activeOutput() {
        Window window = WindowManager.getInstance().suggestParentWindow(project);
        if (window == null) {
            LOG.warn("window is null");
            return;
        }
        OutputService.getInstance(project).active();
        OutputFactory.show(project);
    }

    /**
     * setIcon
     *
     * @param jLabel jLabel
     * @param dark   dark
     * @param light  light
     */
    private void setIcon(JLabel jLabel, Icon dark, Icon light) {
        jLabel.setIcon(UiUtils.getJbIcon(dark, light));
        jLabel.setIconTextGap(ICON_TEXT_GAP);
    }

    /**
     * initComponent
     */
    private void initComponent() {
        root.setVisible(true);
        initMouse();
        root.addPropertyChangeListener(propertyChangeEvent -> {
           initIcon();
           initMap();
        });
    }

    /**
     * initMap
     */
    private void initMap() {
        ModelAnalyse modelAnalyse = new ModelAnalyse();
        ModelConvert modelConvert = new ModelConvert();
        SystemProfiling systemProfiling = new SystemProfiling();
        AisBench aisBench = new AisBench();
        Compare compare = new Compare();
    }

    /**
     * getRoot
     *
     * @return root
     */
    public JComponent getRoot() {
        return root;
    }

    /**
     * initMouse
     */
    private void initMouse() {
        checkPluginsAndAddAction(modelAnalyse, StepNum.MODEL_ANALYSE, PluginClassId.Inference_PluginId);
        checkPluginsAndAddAction(modelConverter, StepNum.MODEL_CONVERTER, PluginClassId.AitIde_PluginId);
        checkPluginsAndAddAction(aisBench, StepNum.AIS_BENCH, PluginClassId.AitIde_PluginId);
        checkPluginsAndAddAction(compare, StepNum.COMPARE, PluginClassId.AitIde_PluginId);
        checkPluginsAndAddAction(systemProfiler, StepNum.SYSTEM_PROFILER, PluginClassId.Profiler_PluginId);
    }

    /**
     * checkPluginsAndAddAction
     *
     * @param jLabel jLabel
     * @param step    step
     * @param pluginClass pluginClass
     */
    private void checkPluginsAndAddAction(JLabel jLabel, StepNum step, String  pluginClass) {
        if (PluginId.findId(pluginClass) != null && PluginManager.getInstance().findEnabledPlugin(PluginId.getId(pluginClass)) != null) {
            jLabel.addMouseListener(new ChooseViewMouseAdapter(step, jLabel));
        }
    }

    private class ModelAnalyse {
        ModelAnalyse() {
            setIcon(modelAnalyse, Icons.MODEL_ANALYSE_DARK, Icons.MODEL_ANALYSE_LIGHT);
            setStepIcons(modelAnalyseJpanel, modelAnalyse, step1, PluginClassId.Inference_PluginId);
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
            setIcon(modelConverter, Icons.AIT_MODEL_CONVERTER_DARK, Icons.AIT_MODEL_CONVERTER_LIGHT);
            setStepIcons(modelConverterJPanel, modelConverter, step2, PluginClassId.AitIde_PluginId);
            actionMappings.put(StepNum.MODEL_CONVERTER, this::doModelConvert);
        }

        private Object doModelConvert(Project project){
            activeOutput();
            PluginGet pluginGet = new PluginGet(project);
            pluginGet.getPluginClass(PluginClassId.AitModelConverter_ClassId, PluginClassId.AitIde_PluginId);
            return Optional.empty();
        }
    }

    private class AisBench {
        AisBench() {
            setIcon(aisBench, Icons.AIS_BENCH_DARK, Icons.AIS_BENCH_LIGHT);
            setStepIcons(aisBenchJPanel, aisBench, step3, PluginClassId.AitIde_PluginId);
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
            setStepIcons(compareJPanel, compare, step4, PluginClassId.AitIde_PluginId);
            actionMappings.put(StepNum.COMPARE, this::doCompare);
        }

        private Object doCompare(Project project){
            activeOutput();
            PluginGet pluginGet = new PluginGet(project);
            pluginGet.getPluginClass(PluginClassId.Compare_ClassId, PluginClassId.AitIde_PluginId);
            return Optional.empty();
        }
    }

    private class SystemProfiling {
        SystemProfiling() {
            setIcon(systemProfiler, Icons.SYSTEM_PROFILER_DARK, Icons.SYSTEM_PROFILER_LIGHT);
            setStepIcons(systemProfilerJPanel, systemProfiler, step5, PluginClassId.Profiler_PluginId);
            actionMappings.put(StepNum.SYSTEM_PROFILER, this::doSystemProfiling);
        }

        private Object doSystemProfiling(Project project){
            activeOutput();
            PluginGet pluginGet = new PluginGet(project);
            pluginGet.getPluginClass(PluginClassId.SystemProfiler_ClassId, PluginClassId.Profiler_PluginId);
            return Optional.empty();
        }
    }

    private void setStepIcons(JPanel jPanel, JLabel jLabel1, JLabel jLabel2, String  pluginClass) {
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
