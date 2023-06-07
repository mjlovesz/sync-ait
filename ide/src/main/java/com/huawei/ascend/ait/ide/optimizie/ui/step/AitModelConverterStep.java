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

import com.huawei.ascend.ait.ide.optimizie.aitmodelconvert.AitModelConvertTask;
import com.huawei.ascend.ait.ide.util.FileChooseWithBrows;
import com.huawei.ascend.ait.ide.util.exception.PathInvalidException;
import com.huawei.ascend.ait.ide.util.exception.ModelFileInvalidException;
import com.intellij.icons.AllIcons;
import com.intellij.openapi.fileChooser.FileChooserDescriptor;
import com.intellij.openapi.progress.ProgressIndicator;
import com.intellij.openapi.progress.ProgressManager;
import com.intellij.openapi.progress.Task;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.ui.DialogWrapper;
import com.intellij.openapi.ui.Messages;
import com.intellij.openapi.ui.TextFieldWithBrowseButton;
import com.intellij.openapi.util.io.FileUtil;
import com.intellij.ui.DocumentAdapter;
import com.intellij.ui.JBColor;
import com.jediterm.terminal.util.JTextFieldLimit;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.lang.StringUtils;
import org.cef.OS;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import javax.swing.JPanel;
import javax.swing.JLabel;
import javax.swing.JComboBox;
import javax.swing.JTextField;
import javax.swing.JComponent;
import javax.swing.BorderFactory;
import javax.swing.border.Border;
import javax.swing.event.DocumentEvent;
import java.awt.Dimension;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.PosixFilePermission;
import java.util.List;
import java.util.MissingResourceException;
import java.util.Objects;
import java.util.Set;

public class AitModelConverterStep extends DialogWrapper {
    private JPanel root;
    private JComboBox<String> socVersionComboBox;
    private TextFieldWithBrowseButton modelFileBrowse;
    private JTextField modelFileTextField = modelFileBrowse.getTextField();
    private Border defBorder = modelFileTextField.getBorder();
    private TextFieldWithBrowseButton outputPathBrowse;
    private JTextField outputPathTextField = outputPathBrowse.getTextField();
    private JPanel outputPathErrPanel;
    private JLabel outputPathErrLabel;
    private JPanel modelFileErrPanel;
    private JLabel modelFileErrLabel;
    private JTextField modelNameTextField;
    private JPanel modelNameErrPanel;
    private JLabel modelNameErrLabel;
    private TextFieldWithBrowseButton cannPathBrowse;
    private JTextField cannPathTextField = cannPathBrowse.getTextField();
    private TextFieldWithBrowseButton aiePathBrowse;
    private JPanel cannPathErrPanel;
    private JLabel cannPathErrLabel;
    private JPanel aiePathErrPanel;
    private JLabel aiePathErrLabel;
    private JTextField aiePathTextField = aiePathBrowse.getTextField();
    private final Project project;
    private static final List<String> SOC_VERSION_LIST = List.of("Ascend310P3", "Ascend910B3");
    private static final long FILE_SIZE_LIMIT_2G = 2 * 1024 * 1024 * 1024L;
    private static final int DOCUMENT_LIMIT = 256;
    private static final int MODEL_NAME_LIMIT = 64;
    private static final String ONNX_MODEL_FILE_SUFFIX = ".onnx";
    private static final String OM_MODEL_FILE_SUFFIX = ".om";

    public AitModelConverterStep(@Nullable Project project) {
        super(true);
        this.project = project;

        init();
        setTitle("Ait Model Converter");

        initComponent();
        initVisible();
        setTextFieldLimitAndToolTip();
        addCannPathBrowseListener();
        addCannPathTextListener();
        addAiePathBrowseListener();
        addAiePathTextListener();
        addModelFileBrowseListener();
        addModelFileTextFieldListener();
        addOutputPathBrowseListener();
        initOutputPathTextListener();
        addModelNameTextListener();
    }

    private void setTextFieldLimitAndToolTip() {
        cannPathTextField.setDocument(new JTextFieldLimit(DOCUMENT_LIMIT));
        cannPathTextField.setToolTipText("Only letters, digits, and the following special characters are allowed:- . _ : \\ /");
        aiePathTextField.setDocument(new JTextFieldLimit(DOCUMENT_LIMIT));
        aiePathTextField.setToolTipText("Only letters, digits, and the following special characters are allowed:- . _ : \\ /");
        modelFileTextField.setDocument(new JTextFieldLimit(DOCUMENT_LIMIT));
        modelFileTextField.setToolTipText("Only letters, digits, and the following special characters are allowed:- . _ : \\ /");
        modelNameTextField.setDocument(new JTextFieldLimit(DOCUMENT_LIMIT));
        modelNameTextField.setToolTipText("Only letters, digits, hyphens(-), end(.) and underscores(_) are allowed.");
        outputPathTextField.setDocument(new JTextFieldLimit(DOCUMENT_LIMIT));
        outputPathTextField.setToolTipText("Only letters, digits, and the following special characters are allowed:- . _ : \\ /");
    }

    private void initComponent() {
        root.setPreferredSize(new Dimension(500, -1));
        for (String soc : SOC_VERSION_LIST) {
            this.socVersionComboBox.addItem(soc);
        }
        this.socVersionComboBox.setSelectedItem(SOC_VERSION_LIST.get(0));
    }

    private void initVisible() {
        socVersionComboBox.setVisible(true);
        cannPathBrowse.setVisible(true);
        cannPathErrPanel.setVisible(false);
        aiePathBrowse.setVisible(true);
        aiePathErrPanel.setVisible(false);
        modelFileBrowse.setVisible(true);
        modelFileErrPanel.setVisible(false);
        outputPathBrowse.setVisible(true);
        outputPathErrPanel.setVisible(false);
        modelNameTextField.setVisible(true);
        modelNameErrPanel.setVisible(false);
    }

    @Override
    protected @Nullable JComponent createCenterPanel() {
        return root;
    }

    private void addCannPathTextListener() {
        cannPathTextField.getDocument().addDocumentListener(new DocumentAdapter() {
            @Override
            protected void textChanged(@NotNull DocumentEvent event) {
                String cannPath= cannPathTextField.getText();
                if (StringUtils.isEmpty(cannPath)) {
                    normalizeInput(cannPathErrPanel, cannPathErrLabel, cannPathTextField);
                    return;
                }

                try {
                    checkPathValid(cannPath);
                    normalizeInput(cannPathErrPanel, cannPathErrLabel, cannPathTextField);
                } catch (PathInvalidException exception) {
                    abnormalInput(cannPathErrPanel, cannPathErrLabel, cannPathTextField, exception.getMessage());
                }
            }
        });
    }

    private void addAiePathTextListener() {
        aiePathTextField.getDocument().addDocumentListener(new DocumentAdapter() {
            @Override
            protected void textChanged(@NotNull DocumentEvent event) {
                String aiePath= aiePathTextField.getText();
                if (StringUtils.isEmpty(aiePath)) {
                    normalizeInput(aiePathErrPanel, aiePathErrLabel, aiePathTextField);
                    return;
                }

                try {
                    checkPathValid(aiePath);
                    normalizeInput(aiePathErrPanel, aiePathErrLabel, aiePathTextField);
                } catch (PathInvalidException exception) {
                    abnormalInput(aiePathErrPanel, aiePathErrLabel, aiePathTextField, exception.getMessage());
                }
            }
        });
    }

    private void initOutputPathTextListener() {
        outputPathTextField.getDocument().addDocumentListener(new DocumentAdapter() {
            @Override
            protected void textChanged(@NotNull DocumentEvent event) {
                String outputPath= outputPathTextField.getText();
                if (StringUtils.isEmpty(outputPath)) {
                    normalizeInput(outputPathErrPanel, outputPathErrLabel, outputPathTextField);
                    return;
                }

                try {
                    checkOutputPathValid(outputPath);
                    normalizeInput(outputPathErrPanel, outputPathErrLabel, outputPathTextField);

                    String modelName = modelNameTextField.getText();
                    if (!StringUtils.isEmpty(modelName)) {
                        String outputFile = FileUtils.getFile(outputPath, modelName + OM_MODEL_FILE_SUFFIX).toString();
                        if (checkFileExist(outputFile)) {
                            abnormalInput(outputPathErrPanel, outputPathErrLabel, outputPathTextField,
                                    "The model already exists in the output path and it will be overlayed.");
                        }
                    }
                } catch (PathInvalidException exception) {
                    abnormalInput(outputPathErrPanel, outputPathErrLabel, outputPathTextField, exception.getMessage());
                }
            }
        });
    }

    private void checkPathValid(@NotNull String path) throws PathInvalidException {
        if (!pathValid(path)) {
            throw new PathInvalidException("Valid file path characters: -, _, :, \\, /, [0-9], [A-Z], [a-z].");
        }

        File outputPathFile = FileUtils.getFile(path);
        if (FileUtils.isSymlink(outputPathFile)) {
            throw new PathInvalidException("The path is a soft link.");
        }
        if (!FileUtil.exists(path)) {
            throw new PathInvalidException("The path does not exist.");
        }
        if (!FileUtils.getFile(path).isDirectory()) {
            throw new PathInvalidException("The path must be a directory.");
        }
    }

    private void checkOutputPathValid(@NotNull String outputPath) throws PathInvalidException {
        checkPathValid(outputPath);

        if (!FileUtils.getFile(outputPath).canWrite()) {
            throw new PathInvalidException("You do not have the permission to create files in the current directory.");
        }
    }

    private void addCannPathBrowseListener() {
        cannPathBrowse.addActionListener(event -> {
            String outputPath = getCannPathBrowseSelectedFile();
            if (!StringUtils.isEmpty(outputPath)) {
                cannPathBrowse.setText(outputPath);
            }
        });
    }

    private String getCannPathBrowseSelectedFile() {
        FileChooserDescriptor fileChooserDescriptor = new FileChooserDescriptor(false, true, false, false,
                false, false)
                .withTitle("Browse for cann path")
                .withDescription("Please select a cann path");
        return FileChooseWithBrows.fileChoosewithBrowse(project, fileChooserDescriptor,
                this.getClass().getName(), "selectedCannPath").orElse("");
    }

    private void addAiePathBrowseListener() {
        aiePathBrowse.addActionListener(event -> {
            String outputPath = getAiePathBrowseSelectedFile();
            if (!StringUtils.isEmpty(outputPath)) {
                aiePathBrowse.setText(outputPath);
            }
        });
    }

    private String getAiePathBrowseSelectedFile() {
        FileChooserDescriptor fileChooserDescriptor = new FileChooserDescriptor(false, true, false, false,
                false, false)
                .withTitle("Browse for Aie path")
                .withDescription("Please select a Aie path");
        return FileChooseWithBrows.fileChoosewithBrowse(project, fileChooserDescriptor,
                this.getClass().getName(), "selectedAiePath").orElse("");
    }

    private void addModelFileBrowseListener() {
        modelFileBrowse.addActionListener(event -> {
            String modelFile = getModelFileBrowseSelectedFile();
            if (!StringUtils.isEmpty(modelFile)) {
                modelFileBrowse.setText(modelFile);
            }
            File file = new File(modelFile);
            if (file.length() > FILE_SIZE_LIMIT_2G) {
                int result = Messages.showDialog("The size of the model file exceed 2 GB, Do you want to continue?",
                        "Please Confirm", new String[] {"Yes", "No"}, Messages.NO, AllIcons.General.QuestionDialog);
                if (result == Messages.NO) {
                    return;
                }
            }
            modelNameTextField.setText(FilenameUtils.getBaseName(modelFile));
        });
    }

    private void addModelFileTextFieldListener() {
        modelFileBrowse.getTextField().getDocument().addDocumentListener(new DocumentAdapter() {
            @Override
            protected void textChanged(@NotNull DocumentEvent event) {
                String modelFile = modelFileTextField.getText();
                if (StringUtils.isEmpty(modelFile)) {
                    normalizeInput(modelFileErrPanel, modelFileErrLabel, modelFileTextField);
                    modelNameTextField.setText("");
                    return;
                }

                try {
                    checkModelFileValid(modelFile);
                    normalizeInput(modelFileErrPanel, modelFileErrLabel, modelFileTextField);
                } catch (ModelFileInvalidException exception) {
                    abnormalInput(modelFileErrPanel, modelFileErrLabel, modelFileTextField, exception.getMessage());
                    modelNameTextField.setText("");
                }
                modelNameTextField.setText(FilenameUtils.getBaseName(modelFile));
            }
        });
    }

    private void checkModelFileValid(@NotNull String modelFile) throws ModelFileInvalidException {
        if (!pathValid(modelFile)) {
            throw new ModelFileInvalidException("Valid file path characters: -, _, :, \\, /, [0-9], [A-Z], [a-z].");
        }
        if (!isSupportModelType(modelFile)) {
            throw new ModelFileInvalidException("Unsupported Model Type.");
        }
        if (!checkFileExist(modelFile)) {
            throw new ModelFileInvalidException("The file path does not exist.");
        }
        if (!FileUtils.getFile(modelFile).canRead()) {
            throw new ModelFileInvalidException("The model file is invalid or you do not have the read permission.");
        }
        if (OS.isLinux() && (isOtherWritableFile(modelFile))) {
            throw new ModelFileInvalidException("Warning: Other users have the write permission on this model file.");
        }
    }

    private boolean pathValid(@NotNull String path) {
        return path.matches("(\\.|\\\\|/|:|_|-|[0-9a-zA-Z])+");
    }

    private boolean isSupportModelType(@NotNull String path) {
        return path.endsWith(ONNX_MODEL_FILE_SUFFIX);
    }

    private boolean checkFileExist(@NotNull String path) {
        String checkFilePath = path.replaceAll("//", "/");
        File file = FileUtils.getFile(checkFilePath);
        try {
            if (!Files.exists(Paths.get(file.getCanonicalPath()))) {
                return false;
            }
        } catch (IOException exception) {
            return false;
        }
        return true;
    }

    private boolean isOtherWritableFile(@NotNull String path) {
        try {
            Set<PosixFilePermission> permissions = Files.getPosixFilePermissions(Path.of(path));
            return permissions.contains(PosixFilePermission.OTHERS_WRITE);
        } catch (IOException exception) {
            return false;
        }
    }

    private void addOutputPathBrowseListener() {
        outputPathBrowse.addActionListener(event -> {
            String outputPath = getOutputPathBrowseSelectedFile();
            if (!StringUtils.isEmpty(outputPath)) {
                outputPathBrowse.setText(outputPath);
            }
        });
    }

    private void addModelNameTextListener() {
        modelNameTextField.getDocument().addDocumentListener(new DocumentAdapter() {
            @Override
            protected void textChanged(@NotNull DocumentEvent event) {
                String modelName = modelNameTextField.getText();
                if (StringUtils.isEmpty(modelName)) {
                    normalizeInput(modelNameErrPanel, modelNameErrLabel, modelNameTextField);
                } else {
                    checkModelNameValid();
                }
            }
        });
    }

    private String getModelFileBrowseSelectedFile() {
        FileChooserDescriptor fileChooserDescriptor = new FileChooserDescriptor(true, false, false, false,
                false, false)
                .withFileFilter(filter -> "onnx".equalsIgnoreCase(filter.getExtension()))
                .withTitle("Browse for Model File")
                .withDescription("Please select a model file");
        return FileChooseWithBrows.fileChoosewithBrowse(project, fileChooserDescriptor,
                this.getClass().getName(), "selectedModelFile").orElse("");
    }

    private String getOutputPathBrowseSelectedFile() {
        FileChooserDescriptor fileChooserDescriptor = new FileChooserDescriptor(false, true, false, false,
                false, false)
                .withTitle("Browse for output path")
                .withDescription("Please select a output path");
        return FileChooseWithBrows.fileChoosewithBrowse(project, fileChooserDescriptor,
                this.getClass().getName(), "selectedOutputPath").orElse("");
    }

    private boolean isModelFileValid() {
        String modelFile = modelFileTextField.getText();
        if (StringUtils.isEmpty(modelFile)) {
            modelFileTextField.setBorder(BorderFactory.createLineBorder(JBColor.RED));
            abnormalInput(modelFileErrPanel, modelFileErrLabel, modelFileTextField,
                    "The input model file is empty.");
            return false;
        }

        try {
            checkModelFileValid(modelFile);
        } catch (ModelFileInvalidException exception) {
            abnormalInput(modelFileErrPanel, modelFileErrLabel, modelFileTextField, exception.getMessage());
            modelNameTextField.setText("");
            return false;
        }

        return true;
    }

    private boolean checkModelNameValid() {
        String modelName = modelNameTextField.getText();
        if (StringUtils.isEmpty(modelName)) {
            modelNameTextField.setBorder(BorderFactory.createLineBorder(JBColor.RED));
            abnormalInput(modelNameErrPanel, modelNameErrLabel, modelNameTextField,
                    "The output model name is empty.");
            return false;
        }

        if (!modelName.matches("(_|\\.|-|[0-9a-zA-Z])+")) {
            abnormalInput(modelNameErrPanel, modelNameErrLabel, modelNameTextField,
                    "Only letters, digits, hyphens(-), end(.) and underscores(_) are allowed.");
            return false;
        }

        if (modelName.length() > MODEL_NAME_LIMIT) {
            abnormalInput(modelNameErrPanel, modelNameErrLabel, modelNameTextField,
                    "The model name can contain up to 64 characters.");
            return false;
        }

        return true;
    }

    private boolean isOutputPathValid() {
        String outputPath = outputPathTextField.getText();
        if (StringUtils.isEmpty(outputPath)) {
            outputPathTextField.setBorder(BorderFactory.createLineBorder(JBColor.RED));
            abnormalInput(outputPathErrPanel, outputPathErrLabel, outputPathTextField,
                    "The output path is empty.");
            return false;
        }
        try {
            checkOutputPathValid(outputPath);
        } catch (PathInvalidException exception) {
            abnormalInput(outputPathErrPanel, outputPathErrLabel, outputPathTextField, exception.getMessage());
            return false;
        }
        return true;
    }

    private boolean isInputParamValid() {
        boolean isValid = true;

        isValid &= isModelFileValid();
        isValid &= checkModelNameValid();
        isValid &= isOutputPathValid();

        return isValid;
    }

    @Override
    protected void doOKAction() {
        super.doOKAction();
        Task.WithResult<Boolean, MissingResourceException> backgroundTask =
            new Task.WithResult<Boolean, MissingResourceException>(project, "Checking", false) {
                @Override
                protected Boolean compute(@NotNull ProgressIndicator indicator) throws MissingResourceException {
                    return isInputParamValid();
                }
            };

        Boolean isInputValid = false;
        try {
            isInputValid = ProgressManager.getInstance().run(backgroundTask);
        } catch (MissingResourceException exception) {
            return;
        }

        if (!isInputValid) {
            return;
        }

        String outputFile = FileUtils.getFile(outputPathTextField.getText(), modelNameTextField.getText() + OM_MODEL_FILE_SUFFIX).toString();
        if (checkFileExist(outputFile)) {
            int ret = Messages.showYesNoDialog("The model already exists in the output directory. Do you want to replace the following model or use another name?"
                                            + System.getProperty("line.separator") + outputPathTextField.getText(),
                                        "Info", "Replace", "Rename",null);
            if (ret != Messages.YES) {
                return;
            }
        }

        ProgressManager.getInstance().run(new AitModelConvertTask(project, cannPathTextField.getText(), aiePathTextField.getText(),
                modelFileTextField.getText(), Objects.requireNonNull(socVersionComboBox.getSelectedItem()).toString(), outputFile));
    }

    private void normalizeInput(@NotNull JPanel panel, @NotNull JLabel label,
                               @NotNull JTextField textField) {
        textField.setBorder(defBorder);
        label.setText("");
        panel.setVisible(false);
    }

    private void abnormalInput(@NotNull JPanel errPanel, @NotNull JLabel errLabel,
                            @NotNull JTextField textField, @NotNull String errMsg) {
        errPanel.setVisible(true);
        errLabel.setText(errMsg);
        errLabel.setForeground(JBColor.RED);
        textField.setBorder(BorderFactory.createLineBorder(JBColor.RED));
    }
}
