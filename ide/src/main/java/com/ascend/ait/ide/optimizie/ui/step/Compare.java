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

package com.ascend.ait.ide.optimizie.ui.step;

import com.ascend.ait.ide.Icons;
import com.ascend.ait.ide.commonlib.exception.CommandInjectException;
import com.ascend.ait.ide.commonlib.output.OutputService;
import com.ascend.ait.ide.commonlib.ui.SwitchButton;
import com.ascend.ait.ide.commonlib.util.safeCmd.CmdExec;
import com.ascend.ait.ide.commonlib.util.safeCmd.CmdStrBuffer;

import com.ascend.ait.ide.service.AisBenchService;
import com.ascend.ait.ide.service.CompareService;
import com.ascend.ait.ide.util.FileChooseWithBrows;
import com.intellij.execution.ui.ConsoleViewContentType;
import com.intellij.openapi.fileChooser.FileChooserDescriptor;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.ui.DialogWrapper;
import com.intellij.openapi.ui.TextFieldWithBrowseButton;
import com.intellij.openapi.util.Comparing;
import org.apache.commons.lang.StringUtils;
import org.jetbrains.annotations.Nullable;

import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Compare extends DialogWrapper {
    private JPanel root;
    private JLabel modelPathJLabel;
    private TextFieldWithBrowseButton modelFileBrowse;
    private JTextField inputShapeJText;
    private JLabel debug;
    private SwitchButton advisorButton;
    private TextFieldWithBrowseButton offlineModelPathBrowse;
    private TextFieldWithBrowseButton inputPathBrowse;
    private TextFieldWithBrowseButton cannPathBrowse;
    private JLabel offlineModelPathJLabel;
    private JLabel inputPathJLabel;
    private JLabel cannPathJLabel;
    private TextFieldWithBrowseButton outputPathBrowse;
    private JTextField deviceJText;
    private SwitchButton dumpButton;
    private SwitchButton convertButton;
    private JTextField dymShapeJtext;
    private JTextField outputSizeJText;
    private final Project project;
    private static final String OM_MODEL_FILE_EXTENSION = "om";

    public Compare(@Nullable Project project) {
        super(true);
        this.project = project;
        init();
        setTitle("Compare");
        setIcons();

        setFileChoodeAction();
        initComponent();
        setOKButtonText("Start");
    }

    private void setIcons() {
        modelPathJLabel.setIcon(Icons.STAR);
        offlineModelPathJLabel.setIcon(Icons.STAR);
    }

    private void setFileChoodeAction() {
        modelFIleAction();
        offlineModelAction();
        inputAction();
        cannPathAction();
        outputAction();
    }

    private void modelFIleAction() {
        List<String> lists = List.of(OM_MODEL_FILE_EXTENSION);
        modelFileBrowse.addActionListener(event -> {
            String selectFile = getSelectedFile(lists, true, false);
            if (StringUtils.isEmpty(selectFile)) {
                return;
            }
            File model = new File(selectFile);
            modelFileBrowse.setText(selectFile);
        });
    }

    private void offlineModelAction() {
        List<String> lists = List.of(OM_MODEL_FILE_EXTENSION);
        offlineModelPathBrowse.addActionListener(event -> {
            String selectFile = getSelectedFile(lists, true, false);
            if (StringUtils.isEmpty(selectFile)) {
                return;
            }
            File model = new File(selectFile);
            offlineModelPathBrowse.setText(selectFile);
        });
    }

    private void inputAction() {
        inputPathBrowse.addActionListener(event -> {
            String selectFile = getSelectedFile(null, false, false);
            if (StringUtils.isEmpty(selectFile)) {
                return;
            }
            File model = new File(selectFile);
            inputPathBrowse.setText(selectFile);
        });
    }

    private void cannPathAction() {
        cannPathBrowse.addActionListener(event -> {
            String selectFile = getSelectedFile(null, false, false);
            if (StringUtils.isEmpty(selectFile)) {
                return;
            }
            File model = new File(selectFile);
            cannPathBrowse.setText(selectFile);
        });
    }
    private void outputAction() {
        outputPathBrowse.addActionListener(event -> {
            String selectFile = getSelectedFile(null, false, false);
            if (StringUtils.isEmpty(selectFile)) {
                return;
            }
            File model = new File(selectFile);
            outputPathBrowse.setText(selectFile);
        });
    }
    private String getSelectedFile(List<String> strings, Boolean isFile, Boolean chooseMultiple) {
        FileChooserDescriptor fileChooserDescriptor = new FileChooserDescriptor(isFile, !isFile, false, false,
                false, chooseMultiple)
                .withFileFilter(virtualFile -> virtualFile.isDirectory() ||
                        Comparing.equal(new ArrayList<>(strings).contains(virtualFile.getExtension()), true))
                .withTitle("Model File")
                .withDescription("Please select the appropriate file");
        return FileChooseWithBrows.fileChoosewithBrowse(project, fileChooserDescriptor,
                this.getClass().getName(), "modelSelectPath").orElse(null);
    }

    private void initComponent() {
    }

    @Override
    protected void doOKAction() {
        CmdStrBuffer cmdStrBuffer = new CmdStrBuffer();
        cmdStrBuffer = getCmdStrBuffer();
        OutputService.getInstance(project).print("python3 " + cmdStrBuffer.toString());
        CmdExec exec = new CmdExec();
        try {
            exec.pythonStart(cmdStrBuffer);
            String execRec = exec.getResult();
            if (execRec != null) {
                OutputService.getInstance(project).print(execRec, ConsoleViewContentType.LOG_INFO_OUTPUT);
            }
        } catch (IOException | CommandInjectException e) {
            throw new RuntimeException(e);
        }
        close(0);
    }
    private CmdStrBuffer getCmdStrBuffer() {
        CmdStrBuffer cmd = new CmdStrBuffer();
        cmd.append(" main.py");
        AisBenchService service = new AisBenchService();

        service.pathAdd(cmd, CompareService.modelService, modelFileBrowse.getText());
        service.pathAdd(cmd, CompareService.inputService, inputPathBrowse.getText());
        service.pathAdd(cmd, CompareService.offlineService, offlineModelPathBrowse.getText());
        service.pathAdd(cmd, CompareService.outputService, outputPathBrowse.getText());
        service.pathAdd(cmd, CompareService.cannService, cannPathBrowse.getText());

        service.strAdd(cmd, CompareService.inputShapeService, inputPathJLabel.getText());
        service.strAdd(cmd, CompareService.dymShapeService, dymShapeJtext.getText());
        service.strAdd(cmd, CompareService.outputNodesService, outputSizeJText.getText());
        service.strAdd(cmd, CompareService.inputService, inputPathJLabel.getText());
        service.strAdd(cmd, CompareService.deviceService, deviceJText.getText());

        service.statueAdd(cmd, CompareService.deviceService, advisorButton.isSelected());
        service.statueAdd(cmd, CompareService.dumpService, dumpButton.isSelected());
        service.statueAdd(cmd, CompareService.convertService, convertButton.isSelected());

        return cmd;
    }
    private Boolean preCheck() {
        return true;
    }

    @Override
    protected @Nullable JComponent createCenterPanel() {
        return root;
    }
}
