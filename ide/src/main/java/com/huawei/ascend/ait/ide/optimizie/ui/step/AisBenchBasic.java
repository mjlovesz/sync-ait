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

package com.huawei.ascend.ait.ide.optimizie.ui.step;

import static com.huawei.ascend.ait.ide.service.AisBenchCmdStr.addPath;
import static com.huawei.ascend.ait.ide.service.AisBenchCmdStr.addState;
import static com.huawei.ascend.ait.ide.util.FileChoose.getSelectedFile;
import static com.huawei.ascend.ait.ide.util.FileChoose.getSelectedPath;

import com.huawei.ascend.ait.ide.Icons;
import com.huawei.ascend.ait.ide.commonlib.exception.CommandInjectException;
import com.huawei.ascend.ait.ide.commonlib.output.OutputService;
import com.huawei.ascend.ait.ide.commonlib.util.safeCmd.CmdExec;
import com.huawei.ascend.ait.ide.commonlib.util.safeCmd.CmdStrBuffer;
import com.huawei.ascend.ait.ide.commonlib.util.safeCmd.CmdStrWordStatic;
import com.huawei.ascend.ait.ide.commonlib.ui.SwitchButton;

import com.intellij.execution.ui.ConsoleViewContentType;
import com.intellij.icons.AllIcons;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.ui.DialogWrapper;
import com.intellij.openapi.ui.Messages;
import com.intellij.openapi.ui.TextFieldWithBrowseButton;

import org.apache.commons.lang.StringUtils;
import org.jetbrains.annotations.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.JToggleButton;
import java.io.File;
import java.io.IOException;
import java.util.List;


/**
 * AisBenchBasic
 *
 * @author cabbage
 * @date 2023/06/03
 */
public class AisBenchBasic extends DialogWrapper {
    private static final Logger LOGGER = LoggerFactory.getLogger(AisBenchBasic.class);
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
    private JPanel advanceOptions;
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

    /**
     * AisBenchBasic
     *
     * @param project project
     */
    public AisBenchBasic(Project project) {
        super(true);
        this.project = project;
        init();
        setTitle("Ais Bench");

        setFileChooseAction();
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
        advanceOptions.setVisible(false);

        pureDataTypeJLabel.setVisible(false);
        pureDataTypeCombx.setVisible(false);
        pureDataTypeCombx.setEditable(false);

        outputDirnameJLabel.setVisible(false);
        outputDirnameBrowse.setVisible(false);

        outFmtJLabel.setVisible(false);
        outFormatComboBox.setVisible(false);
    }

    private void setFileChooseAction() {
        modelFIleAction();
        inputAction();
        outputAction();
    }

    private void modelFIleAction() {
        List<String> lists = List.of(OM_MODEL_FILE_EXTENSION);
        modelFileBrowse.addActionListener(event -> {
            String selectFile = getSelectedFile(project, lists, false);
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
            String selectFile = getSelectedFile(project, lists, true);
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
        outputPathBrowse.addActionListener(event -> {
            String selectFile = getSelectedPath(project);
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
        outputDirnameBrowse.addActionListener(event -> {
            String selectFile = getSelectedPath(project);
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
        }
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
        OutputService.getInstance(project).print(cmdStrBuffer.toString());
        CmdExec exec = new CmdExec();
        try {
            exec.bashStart(cmdStrBuffer);
            String errorRec = exec.getErrorResult();
            close(0);
            if (errorRec != null) {
                OutputService.getInstance(project).print("There are some errors here.", ConsoleViewContentType.LOG_DEBUG_OUTPUT);
                OutputService.getInstance(project).print(errorRec, ConsoleViewContentType.LOG_ERROR_OUTPUT);
            }
            String execRec = exec.getResult();
            if (execRec != null) {
                OutputService.getInstance(project).print(execRec, ConsoleViewContentType.LOG_INFO_OUTPUT);
            }
        } catch (CommandInjectException | IOException e) {
            LOGGER.error(e.getMessage());
        }
    }

    private CmdStrBuffer getCmdStrBuffer() {
        CmdStrBuffer cmd = new CmdStrBuffer();

        cmd.append("python3").append(CmdStrWordStatic.SPACE);
        cmd.append("-m").append(CmdStrWordStatic.SPACE);
        cmd.append("ais_bench").append(CmdStrWordStatic.SPACE);
        addPath(cmd, "--model", modelFileTextField.getText());
        addPath(cmd, "--input", inputFilesTextField.getText());
        addPath(cmd, "--pure", pureDataTypeCombx.getSelectedItem().toString());
        addPath(cmd, "--output", outputTextField.getText());
        addPath(cmd, "--outputdir", outputDirTextField.getText());
        addPath(cmd, "--outfmt", outFormatComboBox.getSelectedItem().toString());
        addPath(cmd, "--loop", loopTextField.getText());
        addPath(cmd, "--warmup_count", warmupTextField.getText());
        addPath(cmd, "--device", deviceTextField.getText());
        addState(cmd, "--debug", debugButton.isSelected());
        addState(cmd, "--display_all_summary", displayButton.isSelected());

        return cmd;
    }

    private Boolean preCheck() {
        String model = modelFileTextField.getText();
        if (model.isEmpty()) {
            Messages.showErrorDialog("Model file must be chose", "ERROR");
            return false;
        }
        return true;
    }
}

