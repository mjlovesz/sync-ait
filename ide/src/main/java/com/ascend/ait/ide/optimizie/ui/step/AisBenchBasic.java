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
import com.ascend.ait.ide.commonlib.ui.SwitchButton;
import com.ascend.ait.ide.service.AisBenchService;
import com.ascend.ait.ide.util.FileChooseWithBrows;
import com.ascend.ait.ide.commonlib.exception.CommandInjectException;
import com.ascend.ait.ide.commonlib.output.OutputService;
import com.ascend.ait.ide.commonlib.util.safeCmd.CmdExec;
import com.ascend.ait.ide.commonlib.util.safeCmd.CmdStrBuffer;
import com.intellij.execution.ui.ConsoleViewContentType;
import com.intellij.icons.AllIcons;
import com.intellij.openapi.fileChooser.FileChooserDescriptor;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.ui.DialogWrapper;
import com.intellij.openapi.ui.Messages;
import com.intellij.openapi.ui.TextFieldWithBrowseButton;
import com.intellij.openapi.util.Comparing;
import org.apache.commons.lang.StringUtils;
import org.jetbrains.annotations.Nullable;

import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.JToggleButton;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class AisBenchBasic extends DialogWrapper {
    private JPanel root;
    private JComboBox pureDataTypeCombx;
    private JLabel model;
    private JComboBox outFormatComboBox;
    private JTextField loopTextField;
    private JTextField warmupTextField;
    private JLabel debug;
    private JLabel dusplay;
    private JPanel loop;
    private JToggleButton profilerBottun;
    private JPanel advance;
    private TextFieldWithBrowseButton modelFileBrowse;
    private TextFieldWithBrowseButton inputFileBrowse;
    private TextFieldWithBrowseButton outputPathBrowse;
    private TextFieldWithBrowseButton outputDirnameBrowse;
    private JPanel pureDataTypeJLabel;
    private JPanel outputDirnameJLabel;
    private JPanel outFmtJLabel;
    private JTextField deviceTextField;
    private SwitchButton debugButton;
    private SwitchButton displayButton;
    private final JTextField modelFileTextField = modelFileBrowse.getTextField();
    private final JTextField inputFilesTextField = inputFileBrowse.getTextField();
    private final JTextField outputTextField = outputPathBrowse.getTextField();
    private final JTextField outputDirTextField = outputDirnameBrowse.getTextField();
    private static final String OM_MODEL_FILE_EXTENSION = "om";
    private static final String NPY_FILE_EXTENSION = "npy";
    private static final String BIN_FILE_EXTENSION = "bin";
    private static final String TXT_FILE_EXTENSION = "txt";
    private static final List<String> PURE_DATA_TYPE = List.of("zero", "random");
    private static final List<String> OUTFMT_TYPE = List.of("BIN", "NPY", "TXT");
    private static final long FILE_SIZE_MAX = (long) 2 * 1024 * 1024 * 1024;
    private final Project project;

    public AisBenchBasic(Project project) {
        super(true);
        this.project = project;
        init();
        setTitle("Ais Bench");
        setIcons();

        setFileChoodeAction();
        initComponent();
        initVisible();
        setOKButtonText("Start");
    }


    private void initComponent() {
        for (String s : PURE_DATA_TYPE) {
            this.pureDataTypeCombx.addItem(s);
        }
        for (String s : OUTFMT_TYPE) {
            this.outFormatComboBox.addItem(s);
        }
    }

    private void initVisible() {
        advance.setVisible(false);

        pureDataTypeJLabel.setVisible(false);
        pureDataTypeCombx.setVisible(false);
        pureDataTypeCombx.setEditable(false);

        outputDirnameJLabel.setVisible(false);
        outputDirnameBrowse.setVisible(false);

        outFmtJLabel.setVisible(false);
        outFormatComboBox.setVisible(false);
    }

    private void setIcons() {
        model.setIcon(Icons.STAR);
    }

    private void setFileChoodeAction() {
        modelFIleAction();
        inputAction();
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
            checkFileSize(model);
            modelFileTextField.setText(selectFile);

        });
    }

    private void inputAction() {
        List<String> lists = List.of(NPY_FILE_EXTENSION, BIN_FILE_EXTENSION);
        inputFileBrowse.addActionListener(event -> {
            String selectFile = getSelectedFile(lists, true, true);
            if (StringUtils.isEmpty(selectFile)) {
                return;
            }
            File model = new File(selectFile);
            checkFileSize(model);
            inputFilesTextField.setText(selectFile);
            pureDataTypeJLabel.setVisible(true);
            pureDataTypeCombx.setVisible(true);
        });
    }

    private void outputAction() {
        List<String> lists = List.of(NPY_FILE_EXTENSION, BIN_FILE_EXTENSION);
        outputPathBrowse.addActionListener(event -> {
            String selectFile = getSelectedFile(lists, false, false);
            if (StringUtils.isEmpty(selectFile)) {
                return;
            }
            File model = new File(selectFile);
            checkFileSize(model);
            outputTextField.setText(selectFile);

            outputDirnameBrowse.setVisible(true);
            outputDirnameJLabel.setVisible(true);

            outFmtJLabel.setVisible(true);
            outFormatComboBox.setVisible(true);
            outputdIRAction();
        });
    }

    private void outputdIRAction() {
        List<String> lists = List.of(NPY_FILE_EXTENSION, BIN_FILE_EXTENSION);
        outputDirnameBrowse.addActionListener(event -> {
            String selectFile = getSelectedFile(lists, false, false);
            if (StringUtils.isEmpty(selectFile)) {
                return;
            }
            File model = new File(selectFile);
            checkFileSize(model);
            outputDirTextField.setText(selectFile);
        });
    }

    private void checkFileSize(File file) {
        if (file.length() > FILE_SIZE_MAX) {
            int result = Messages.showDialog("The file you selected is too large.", "ERROR", new String[]{"Yes", "No"},
                    Messages.NO, AllIcons.General.QuestionDialog);
            if (result == Messages.NO) {
                return;
            }
        }
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

    @Override
    protected @Nullable JComponent createCenterPanel() {
        return root;
    }

    @Override
    protected void doOKAction() {
        Boolean check = preCheck();
        if (!check) {
            return;
        }
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
        cmd.append("-m ais_bench");
        AisBenchService service = new AisBenchService();

        service.pathAdd(cmd, AisBenchService.modelService, modelFileTextField.getText());
        service.pathAdd(cmd, AisBenchService.inputService, inputFilesTextField.getText());
        service.strAdd(cmd, AisBenchService.pureService, pureDataTypeCombx.getSelectedItem().toString());
        service.pathAdd(cmd, AisBenchService.outputService, outputTextField.getText());
        service.pathAdd(cmd, AisBenchService.outputdirService, outputDirTextField.getText());
        service.strAdd(cmd, AisBenchService.outfmtService, outFormatComboBox.getSelectedItem().toString());
        service.strAdd(cmd, AisBenchService.loopService, loopTextField.getText());
        service.strAdd(cmd, AisBenchService.warmupService, warmupTextField.getText());
        service.strAdd(cmd, AisBenchService.deviceService, deviceTextField.getText());
        service.statueAdd(cmd, AisBenchService.debugService, debugButton.isSelected());
        service.statueAdd(cmd, AisBenchService.displayService, displayButton.isSelected());

        return cmd;
    }

    private Boolean preCheck() {
        String modelfile = modelFileTextField.getText();
        if (modelfile.isEmpty()) {
            Messages.showErrorDialog("Model file must be chose", "ERROR");
            return false;
        }
        return true;
    }

}

