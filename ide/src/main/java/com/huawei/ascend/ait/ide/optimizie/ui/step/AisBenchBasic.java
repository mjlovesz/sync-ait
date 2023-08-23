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

import static com.huawei.ascend.ait.ide.service.AisBenchCmdStr.add;
import static com.huawei.ascend.ait.ide.service.AisBenchCmdStr.addState;
import static com.huawei.ascend.ait.ide.util.CheckInput.VALID_DIGITS_CHARATERS;
import static com.huawei.ascend.ait.ide.util.CheckInput.checkDigitValid;
import static com.huawei.ascend.ait.ide.util.FileChoose.getSelectedFile;
import static com.huawei.ascend.ait.ide.util.FileChoose.getSelectedPath;
import static com.huawei.ascend.ait.ide.util.CheckInput.VALID_STRING_CHARATERS;
import static com.huawei.ascend.ait.ide.util.CheckInput.checkStringValid;
import static com.huawei.ascend.ait.ide.util.CheckInput.normalizeInput;

import com.huawei.ascend.ait.ide.commonlib.exception.CommandInjectException;
import com.huawei.ascend.ait.ide.commonlib.output.OutputService;
import com.huawei.ascend.ait.ide.commonlib.util.safeCmd.CmdExec;
import com.huawei.ascend.ait.ide.commonlib.util.safeCmd.CmdStrBuffer;
import com.huawei.ascend.ait.ide.commonlib.util.safeCmd.CmdStrWordStatic;
import com.huawei.ascend.ait.ide.commonlib.ui.SwitchButton;
import com.huawei.ascend.ait.ide.util.CheckInput;

import com.intellij.execution.ui.ConsoleViewContentType;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.ui.DialogWrapper;
import com.intellij.openapi.ui.Messages;
import com.intellij.openapi.ui.TextFieldWithBrowseButton;

import com.intellij.ui.DocumentAdapter;
import com.jediterm.terminal.util.JTextFieldLimit;
import org.apache.commons.lang.StringUtils;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.border.Border;
import javax.swing.event.DocumentEvent;
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
    private JTextField loopJText;
    private JTextField warmupJText;
    private JLabel debug;
    private JLabel dusplay;
    private TextFieldWithBrowseButton modelFileBrowse;
    private TextFieldWithBrowseButton inputFileBrowse;
    private TextFieldWithBrowseButton outputPathBrowse;
    private JPanel pureDataTypeJLabel;
    private JPanel outputDirnameJLabel;
    private JPanel outFmtJLabel;
    private JTextField deviceTextField;
    private SwitchButton debugButton;
    private SwitchButton displayButton;
    private JLabel outputDirErrorJLabel;
    private JLabel countErrorJLabel;
    private JLabel deviceErrorJLabel;
    private JTextField outputDirJText;
    private JPanel countErrorJPanel;
    private JPanel outputDirErrorJPanel;
    private JPanel deviceErrorJPanel;
    private final JTextField modelFileTextField = modelFileBrowse.getTextField();
    private final JTextField inputFilesTextField = inputFileBrowse.getTextField();
    private final JTextField outputTextField = outputPathBrowse.getTextField();
    private static final String OM_MODEL_FILE_EXTENSION = "om";
    private static final String NPY_FILE_EXTENSION = "npy";
    private static final String BIN_FILE_EXTENSION = "bin";
    private static final String TXT_FILE_EXTENSION = "txt";
    private static final List<String> PURE_DATA_TYPE = List.of("zero", "random");
    private static final List<String> OUTFMT_TYPE = List.of("BIN", "NPY", "TXT");
    private final Border defBorder = modelFileTextField.getBorder();
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

        setTextFieldLimitAndToolTip();
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
        pureDataTypeJLabel.setVisible(false);
        pureDataTypeCombx.setVisible(false);
        pureDataTypeCombx.setEditable(false);

        outputDirnameJLabel.setVisible(false);
        outputDirJText.setVisible(false);

        outFmtJLabel.setVisible(false);
        outFormatComboBox.setVisible(false);
    }

    private void setFileChooseAction() {
        modelFIleAction();
        inputAction();
        outputAction();
        addOutputDirTextListener();
        addWarmupTextListener();
        addLoopTextListener();
        addDeviceTextListener();
    }

    private void modelFIleAction() {
        List<String> lists = List.of(OM_MODEL_FILE_EXTENSION);
        modelFileBrowse.addActionListener(event -> {
            String selectFile = getSelectedFile(project, lists, false, "Model File Path");
            if (StringUtils.isEmpty(selectFile)) {
                return;
            }
            File model = new File(selectFile);
            if (CheckInput.checkFileSize(model) == Messages.NO) {
                return;
            }
            modelFileTextField.setText(selectFile);
        });
    }

    private void inputAction() {
        List<String> lists = List.of(NPY_FILE_EXTENSION, BIN_FILE_EXTENSION);
        inputFileBrowse.addActionListener(event -> {
            String selectFile = getSelectedFile(project, lists, true, "Input Path");
            if (StringUtils.isEmpty(selectFile)) {
                return;
            }
            String[] files = selectFile.split(",");
            for (String file : files) {
                File f = new File(file);
                if (CheckInput.checkFileSize(f) == Messages.NO) {
                    return;
                }
            }
            inputFilesTextField.setText(selectFile);
            pureDataTypeJLabel.setVisible(true);
            pureDataTypeCombx.setVisible(true);
        });
    }

    private void outputAction() {
        outputPathBrowse.addActionListener(event -> {
            String selectFile = getSelectedPath(project, "Output Path");
            if (StringUtils.isEmpty(selectFile)) {
                return;
            }
            outputTextField.setText(selectFile);

            outputDirJText.setVisible(true);
            outputDirnameJLabel.setVisible(true);

            outFmtJLabel.setVisible(true);
            outFormatComboBox.setVisible(true);
        });
    }

    private void setTextFieldLimitAndToolTip() {
        loopJText.setToolTipText(VALID_DIGITS_CHARATERS);
        loopJText.setDocument(new JTextFieldLimit(255));

        warmupJText.setToolTipText(VALID_DIGITS_CHARATERS);
        warmupJText.setDocument(new JTextFieldLimit(255));

        deviceTextField.setToolTipText(VALID_STRING_CHARATERS);
        deviceTextField.setDocument(new JTextFieldLimit(255));
    }

    private void addOutputDirTextListener() {
        outputDirJText.getDocument().addDocumentListener(new DocumentAdapter() {
            @Override
            protected void textChanged(@NotNull DocumentEvent event) {
                String outputDirText = outputDirJText.getText();
                if (StringUtils.isEmpty(outputDirText)) {
                    normalizeInput(outputDirErrorJPanel, outputDirErrorJLabel, outputDirJText);
                } else {
                    if (checkStringValid(outputDirErrorJPanel, outputDirErrorJLabel, outputDirJText)) {
                        normalizeInput(outputDirErrorJPanel, outputDirErrorJLabel, outputDirJText);
                    }
                }
            }
        });
    }

    private void addLoopTextListener() {
        loopJText.getDocument().addDocumentListener(new DocumentAdapter() {
            @Override
            protected void textChanged(@NotNull DocumentEvent event) {
                String loopText = loopJText.getText();
                if (StringUtils.isEmpty(loopText)) {
                    normalizeInput(countErrorJPanel, countErrorJLabel, loopJText);
                } else {
                    if (checkDigitValid(countErrorJPanel, countErrorJLabel, loopJText)) {
                        normalizeInput(countErrorJPanel, countErrorJLabel, loopJText);
                    }
                }
            }
        });
    }

    private void addWarmupTextListener() {
        warmupJText.getDocument().addDocumentListener(new DocumentAdapter() {
            @Override
            protected void textChanged(@NotNull DocumentEvent event) {
                String modelName = warmupJText.getText();
                if (StringUtils.isEmpty(modelName)) {
                    normalizeInput(countErrorJPanel, countErrorJLabel, warmupJText);
                } else {
                    if (checkDigitValid(countErrorJPanel, countErrorJLabel, warmupJText)) {
                        normalizeInput(countErrorJPanel, countErrorJLabel, warmupJText);
                    }
                }
            }
        });
    }

    private void addDeviceTextListener() {
        deviceTextField.getDocument().addDocumentListener(new DocumentAdapter() {
            @Override
            protected void textChanged(@NotNull DocumentEvent event) {
                String modelName = deviceTextField.getText();
                if (StringUtils.isEmpty(modelName)) {
                    normalizeInput(deviceErrorJPanel, deviceErrorJLabel, deviceTextField);
                } else {
                    if (checkStringValid(deviceErrorJPanel, deviceErrorJLabel, deviceTextField)) {
                        normalizeInput(deviceErrorJPanel, deviceErrorJLabel, deviceTextField);
                    }
                }
            }
        });
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
        close(0);
        try {
            exec.bashStart(cmdStrBuffer);
            String errorRec = exec.getErrorResult();
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
            OutputService.getInstance(project).print(e.getMessage());
        }
    }

    private CmdStrBuffer getCmdStrBuffer() {
        CmdStrBuffer cmd = new CmdStrBuffer();

        cmd.append("python3").append(CmdStrWordStatic.SPACE);
        cmd.append("-m").append(CmdStrWordStatic.SPACE);
        cmd.append("ais_bench").append(CmdStrWordStatic.SPACE);
        add(cmd, "--model", modelFileTextField.getText());
        add(cmd, "--input", inputFilesTextField.getText());
        add(cmd, "--pure", pureDataTypeCombx.getSelectedItem().toString());
        add(cmd, "--output", outputTextField.getText());
        add(cmd, "--output_dirname", outputDirJText.getText());
        add(cmd, "--outfmt", outFormatComboBox.getSelectedItem().toString());
        add(cmd, "--loop", loopJText.getText());
        add(cmd, "--warmup_count", warmupJText.getText());
        add(cmd, "--device", deviceTextField.getText());
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

