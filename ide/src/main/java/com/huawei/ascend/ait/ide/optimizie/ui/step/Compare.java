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

import com.huawei.ascend.ait.ide.commonlib.exception.CommandInjectException;
import com.huawei.ascend.ait.ide.commonlib.output.OutputService;
import com.huawei.ascend.ait.ide.commonlib.util.safeCmd.CmdExec;
import com.huawei.ascend.ait.ide.commonlib.ui.SwitchButton;
import com.huawei.ascend.ait.ide.commonlib.util.safeCmd.CmdStrBuffer;
import com.huawei.ascend.ait.ide.commonlib.util.safeCmd.CmdStrWordStatic;

import com.huawei.ascend.ait.ide.util.FileChooseWithBrows;
import com.intellij.execution.ui.ConsoleViewContentType;
import com.intellij.openapi.fileChooser.FileChooserDescriptor;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.ui.DialogWrapper;
import com.intellij.openapi.ui.Messages;
import com.intellij.openapi.ui.TextFieldWithBrowseButton;

import org.apache.commons.lang.StringUtils;
import org.jetbrains.annotations.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import java.awt.Dimension;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Compare
 *
 * @author cabbage
 * @date 2023/06/03
 */
public class Compare extends DialogWrapper {
    private static final Logger LOGGER = LoggerFactory.getLogger(Compare.class);
    private JPanel root;
    private JLabel model;
    private TextFieldWithBrowseButton modelFileBrowse;
    private JTextField loopTextField;
    private TextFieldWithBrowseButton offlineModelPathBrowse;
    private TextFieldWithBrowseButton inputPathBrowse;
    private TextFieldWithBrowseButton cannPathBrowse;
    private TextFieldWithBrowseButton outputPathBrowse;
    private JTextField inputShapeJText;
    private JTextField deviceJText;
    private JTextField dymShapeJtext;
    private JTextField outputSizeJText;
    private SwitchButton dumpButton;
    private SwitchButton convertButton;
    private JLabel modelPathJLabel;
    private JLabel offlineModelPathJLabel;
    private JLabel inputPathJLabel;
    private JLabel cannPathJLabel;
    private JTextField outputNodesJText;
    private TextFieldWithBrowseButton CompareMainBrowse;
    private SwitchButton debugButton;
    private final Project project;

    private static final String OM_MODEL_FILE_EXTENSION = "om";
    private static final String PB_MODEL_FILE_EXTENSION = "pb";
    private static final String ONNX_MODEL_FILE_EXTENSION = "onnx";

    /**
     * compare
     *
     * @param project project
     */
    public Compare(@Nullable Project project) {
        super(true);
        this.project = project;
        root.setPreferredSize(new Dimension(500, 350));
        dumpButton.setOn();
        init();
        setTitle("Compare");

        setFileChoodeAction();
        setOKButtonText("Start");
    }

    private void setFileChoodeAction() {
        modelFIleAction();
        offlineModelAction();
        inputAction();
        cannPathAction();
        outputAction();
    }

    private void modelFIleAction() {
        List<String> lists = List.of(PB_MODEL_FILE_EXTENSION, ONNX_MODEL_FILE_EXTENSION);
        modelFileBrowse.addActionListener(event -> {
            String selectFile = getSelectedFile(project, lists, false);
            if (StringUtils.isEmpty(selectFile)) {
                return;
            }
            modelFileBrowse.setText(selectFile);
        });
    }

    private void offlineModelAction() {
        List<String> lists = List.of(OM_MODEL_FILE_EXTENSION);
        offlineModelPathBrowse.addActionListener(event -> {
            String selectFile = getSelectedFile(project, lists, false);
            if (StringUtils.isEmpty(selectFile)) {
                return;
            }
            offlineModelPathBrowse.setText(selectFile);
        });
    }

    private void inputAction() {
        inputPathBrowse.addActionListener(event -> {
            FileChooserDescriptor fileChooserDescriptor = new FileChooserDescriptor(true, true, false, false,
                    false, true)
                    .withFileFilter(virtualFile -> virtualFile.isDirectory() || ("bin").equals(virtualFile.getExtension()))
                    .withTitle("Browse for File or Path")
                    .withDescription("Please select the appropriate file of .bin or the path of the file.");
            String selectFile = FileChooseWithBrows.fileChoosewithBrowse(project, fileChooserDescriptor,
                    "", "SelectFile").orElse(null);
            if (StringUtils.isEmpty(selectFile)) {
                return;
            }
            inputPathBrowse.setText(selectFile);
        });
    }

    private void cannPathAction() {
        cannPathBrowse.addActionListener(event -> {
            String selectFile = getSelectedPath(project);
            if (StringUtils.isEmpty(selectFile)) {
                return;
            }
            cannPathBrowse.setText(selectFile);
        });
    }

    private void outputAction() {
        outputPathBrowse.addActionListener(event -> {
            String selectFile = getSelectedPath(project);
            if (StringUtils.isEmpty(selectFile)) {
                return;
            }
            outputPathBrowse.setText(selectFile);
        });
    }

    @Override
    protected void doOKAction() {
        boolean check = preCheck();
        if (!check) {
            return;
        }

        CmdStrBuffer cmdStrBuffer = new CmdStrBuffer();
        cmdStrBuffer = getCmdStrBuffer();
        OutputService.getInstance(project).print(cmdStrBuffer.toString());
        close(0);
        CmdExec exec = new CmdExec();
        try {
            exec.bashStart(cmdStrBuffer);
            String errorRec = exec.getErrorResult();
            if (errorRec != null) {
                OutputService.getInstance(project).print(errorRec, ConsoleViewContentType.LOG_ERROR_OUTPUT);
            }
            String execRec = exec.getResult();
            if (execRec != null) {
                OutputService.getInstance(project).print(execRec, ConsoleViewContentType.LOG_INFO_OUTPUT);
                showCsv();
            }
        } catch (IOException | CommandInjectException e) {
            LOGGER.error(e.getMessage());
            OutputService.getInstance(project).print(e.getMessage());
        }
    }

    private void showCsv() {
        File path = new File(outputPathBrowse.getText());
        File[] files = path.listFiles();
        if (files == null) {
            return;
        }
        Arrays.sort(files, new Comparator<File>() {
            @Override
            public int compare(File file1, File file2) {
                return (int) (file2.lastModified() - file1.lastModified());
            }
        });

        File[] subFiles = files[0].listFiles();
        if (null != subFiles) {
            for (File subFile : subFiles) {
                if (subFile.isFile() && subFile.getName().endsWith(".csv")) {
                    ShowResult showResulte = new ShowResult(project, subFile.getPath());
                    showResulte.show();
                    return;
                }
            }
        }
    }

    private CmdStrBuffer getCmdStrBuffer() {
        CmdStrBuffer cmd = new CmdStrBuffer();

        cmd.append("ait").append(CmdStrWordStatic.SPACE)
                .append("debug").append(CmdStrWordStatic.SPACE)
                .append("compare").append(CmdStrWordStatic.SPACE);
        addPath(cmd, "-gm", modelFileBrowse.getText());
        addPath(cmd, "-om", offlineModelPathBrowse.getText());
        addPath(cmd, "-i", inputPathBrowse.getText());
        addPath(cmd, "-o", outputPathBrowse.getText());
        addPath(cmd, "-c", cannPathBrowse.getText());

        addPath(cmd, "-s", inputShapeJText.getText());
        addPath(cmd, "-dr", dymShapeJtext.getText());
        addPath(cmd, "--output-nodes", outputNodesJText.getText());
        addPath(cmd, "--output-size", outputSizeJText.getText());
        addPath(cmd, "-d", deviceJText.getText());

        addState(cmd, "--dump", dumpButton.isSelected());
        addState(cmd, "--convert", convertButton.isSelected());

        cmd.append("--advisor");

        return cmd;
    }

    private Boolean preCheck() {
        String model = modelFileBrowse.getText();
        String offline = offlineModelPathBrowse.getText();
        String cannPath = cannPathBrowse.getText();
        String output = outputPathBrowse.getText();
        if (model.isEmpty()) {
            Messages.showErrorDialog("Model file must be chose", "ERROR");
            return false;
        }
        if (offline.isEmpty()) {
            Messages.showErrorDialog("Offline model must be chose", "ERROR");
            return false;
        }
        if (cannPath.isEmpty()) {
            Messages.showErrorDialog("CANN path must be chose", "ERROR");
            return false;
        }
        if (output.isEmpty()) {
            Messages.showErrorDialog("Output path must be chose", "ERROR");
            return false;
        }
        return true;
    }

    @Override
    protected @Nullable JComponent createCenterPanel() {
        return root;
    }
}
