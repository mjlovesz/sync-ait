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

import com.huawei.ascend.ait.ide.optimizie.task.AitModelConvertTask;
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

/**
 * AitModelConverterStep
 *
 * @author Jinhaiyang
 * @since 2023/06/03
 */
public class AitModelConverterStep extends DialogWrapper {
    public static final String VALID_DIR_PATH_CHARACTERS = "Valid folder path characters: -, _, :, \\, /, [0-9], [A-Z], [a-z].";
    public static final String UNSUPPORTED_MODEL_TYPE = "Unsupported Model Type.";
    public static final String MODEL_FILE_NOT_EXIST = "The model file does not exist.";
    public static final String VALID_FILE_PATH_CHARATERS = "Only letters, digits, and the following special characters are allowed:- . _ : \\ /";
    public static final String VALID_FILE_NAME_CHARACTERS = "Only letters, digits, hyphens(-), end(.) and underscores(_) are allowed.";
    public static final String MODEL_FILE_ALREADY_EXISTS = "The model already exists in the output path and it will be overlayed.";
    public static final String SOFT_LINK_PATH = "The path is a soft link.";
    public static final String PATH_NOT_EXIST = "The path does not exist.";
    public static final String PATH_MUST_BE_A_DIRECTORY = "The path must be a directory.";
    public static final String PERMISSION_TO_CREATE_FILES = "You do not have the permission to create files in the current directory.";
    public static final String BROWSE_FOR_CANN_PATH = "Browse for CANN path";
    public static final String SELECT_A_CANN_PATH = "Please select a CANN path";
    public static final String SELECTED_CANN_PATH_KEY = "selectedCannPath";
    public static final String BROWSE_FOR_AIE_PATH = "Browse for AIE path";
    public static final String PLEASE_SELECT_A_AIE_PATH = "Please select a AIE path";
    public static final String SELECTED_AIE_PATH_KEY = "selectedAiePath";
    public static final String OTHER_USERS_PERMISSION_INVALID = "Warning: Other users have the write permission on this model file.";
    public static final String NO_READ_PERMISSION = "You do not have the read permission.";
    public static final String PATH_REGULAR_EXPRESSION_PATTERN = "(\\.|\\\\|/|:|_|-|[0-9a-zA-Z])+";
    public static final String BROWSE_FOR_MODEL_FILE = "Browse for Model File";
    public static final String MODEL_FILE_ONNX = "onnx";
    public static final String PLEASE_SELECT_A_MODEL_FILE = "Please select a model file";
    public static final String SELECTED_MODEL_FILE_KEY = "selectedModelFile";
    public static final String BROWSE_FOR_OUTPUT_PATH = "Browse for output path";
    public static final String PLEASE_SELECT_A_OUTPUT_PATH = "Please select a output path";
    public static final String SELECTED_OUTPUT_PATH_KEY = "selectedOutputPath";
    public static final String CANN_PATH_IS_EMPTY = "The CANN path is empty.";
    public static final String AIE_PATH_IS_EMPTY = "The AIE path is empty.";
    public static final String INPUT_MODEL_FILE_IS_EMPTY = "The input model file is empty.";
    public static final String OUTPUT_MODEL_NAME_IS_EMPTY = "The output model name is empty.";
    public static final String MODEL_NAME_MAXIMUM = "The model name can contain up to 64 characters.";
    public static final String OUTPUT_PATH_IS_EMPTY = "The output path is empty.";
    public static final String REPLACE_THE_FOLLOWING_MODEL_OR_USE_ANOTHER_NAME = "The model already exists in the output directory. Do you want to replace the following model or use another name?";
    public static final String FILE_NAME_REGULAR_EXPRESSION_PATTERN = "(_|\\.|-|[0-9a-zA-Z])+";
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
    private static final List<String> SOC_VERSION_LIST = List.of("Ascend310P3");
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
        addOutputPathTextListener();
        addModelNameTextListener();
    }

    private void setTextFieldLimitAndToolTip() {
        cannPathTextField.setDocument(new JTextFieldLimit(DOCUMENT_LIMIT));
        cannPathTextField.setToolTipText(VALID_DIR_PATH_CHARACTERS);
        aiePathTextField.setDocument(new JTextFieldLimit(DOCUMENT_LIMIT));
        aiePathTextField.setToolTipText(VALID_DIR_PATH_CHARACTERS);
        modelFileTextField.setDocument(new JTextFieldLimit(DOCUMENT_LIMIT));
        modelFileTextField.setToolTipText(VALID_FILE_PATH_CHARATERS);
        modelNameTextField.setDocument(new JTextFieldLimit(DOCUMENT_LIMIT));
        modelNameTextField.setToolTipText(VALID_FILE_NAME_CHARACTERS);
        outputPathTextField.setDocument(new JTextFieldLimit(DOCUMENT_LIMIT));
        outputPathTextField.setToolTipText(VALID_DIR_PATH_CHARACTERS);
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

    private void addOutputPathTextListener() {
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
                                    MODEL_FILE_ALREADY_EXISTS);
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
            throw new PathInvalidException(VALID_DIR_PATH_CHARACTERS);
        }

        File outputPathFile = FileUtils.getFile(path);
        if (FileUtils.isSymlink(outputPathFile)) {
            throw new PathInvalidException(SOFT_LINK_PATH);
        }
        if (!FileUtil.exists(path)) {
            throw new PathInvalidException(PATH_NOT_EXIST);
        }
        if (!FileUtils.getFile(path).isDirectory()) {
            throw new PathInvalidException(PATH_MUST_BE_A_DIRECTORY);
        }
    }

    private void checkOutputPathValid(@NotNull String outputPath) throws PathInvalidException {
        checkPathValid(outputPath);

        if (!FileUtils.getFile(outputPath).canWrite()) {
            throw new PathInvalidException(PERMISSION_TO_CREATE_FILES);
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
        FileChooserDescriptor fileChooserDescriptor = new FileChooserDescriptor(false, true,
                false, false,false, false)
                .withTitle(BROWSE_FOR_CANN_PATH)
                .withDescription(SELECT_A_CANN_PATH);
        return FileChooseWithBrows.fileChoosewithBrowse(project, fileChooserDescriptor,
                this.getClass().getName(), SELECTED_CANN_PATH_KEY).orElse("");
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
        FileChooserDescriptor fileChooserDescriptor = new FileChooserDescriptor(false, true,
                false, false,false, false)
                .withTitle(BROWSE_FOR_AIE_PATH)
                .withDescription(PLEASE_SELECT_A_AIE_PATH);
        return FileChooseWithBrows.fileChoosewithBrowse(project, fileChooserDescriptor,
                this.getClass().getName(), SELECTED_AIE_PATH_KEY).orElse("");
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
            throw new ModelFileInvalidException(VALID_DIR_PATH_CHARACTERS);
        }
        if (!isSupportModelType(modelFile)) {
            throw new ModelFileInvalidException(UNSUPPORTED_MODEL_TYPE);
        }
        if (!checkFileExist(modelFile)) {
            throw new ModelFileInvalidException(MODEL_FILE_NOT_EXIST);
        }
        if (!FileUtils.getFile(modelFile).canRead()) {
            throw new ModelFileInvalidException(NO_READ_PERMISSION);
        }
        if (OS.isLinux() && (isOtherWritableFile(modelFile))) {
            throw new ModelFileInvalidException(OTHER_USERS_PERMISSION_INVALID);
        }
    }

    private boolean pathValid(@NotNull String path) {
        return path.matches(PATH_REGULAR_EXPRESSION_PATTERN);
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
                    if (checkModelNameValid()) {
                        normalizeInput(modelNameErrPanel, modelNameErrLabel, modelNameTextField);
                    }
                }
            }
        });
    }

    private String getModelFileBrowseSelectedFile() {
        FileChooserDescriptor fileChooserDescriptor = new FileChooserDescriptor(true, false,
                false, false, false, false)
                .withFileFilter(filter -> MODEL_FILE_ONNX.equalsIgnoreCase(filter.getExtension()))
                .withTitle(BROWSE_FOR_MODEL_FILE)
                .withDescription(PLEASE_SELECT_A_MODEL_FILE);
        return FileChooseWithBrows.fileChoosewithBrowse(project, fileChooserDescriptor,
                this.getClass().getName(), SELECTED_MODEL_FILE_KEY).orElse("");
    }

    private String getOutputPathBrowseSelectedFile() {
        FileChooserDescriptor fileChooserDescriptor = new FileChooserDescriptor(false, true,
                false, false, false, false)
                .withTitle(BROWSE_FOR_OUTPUT_PATH)
                .withDescription(PLEASE_SELECT_A_OUTPUT_PATH);
        return FileChooseWithBrows.fileChoosewithBrowse(project, fileChooserDescriptor,
                this.getClass().getName(), SELECTED_OUTPUT_PATH_KEY).orElse("");
    }

    private boolean isCannPathValid() {
        String cannPath = cannPathTextField.getText();
        if (StringUtils.isEmpty(cannPath)) {
            cannPathTextField.setBorder(BorderFactory.createLineBorder(JBColor.RED));
            abnormalInput(cannPathErrPanel, cannPathErrLabel, cannPathTextField,
                    CANN_PATH_IS_EMPTY);
            return false;
        }

        try {
            checkPathValid(cannPath);
        } catch (PathInvalidException exception) {
            abnormalInput(cannPathErrPanel, cannPathErrLabel, cannPathTextField, exception.getMessage());
            return false;
        }

        return true;
    }

    private boolean isAiePathValid() {
        String cannPath = aiePathTextField.getText();
        if (StringUtils.isEmpty(cannPath)) {
            aiePathTextField.setBorder(BorderFactory.createLineBorder(JBColor.RED));
            abnormalInput(aiePathErrPanel, aiePathErrLabel, aiePathTextField,
                    AIE_PATH_IS_EMPTY);
            return false;
        }

        try {
            checkPathValid(cannPath);
        } catch (PathInvalidException exception) {
            abnormalInput(aiePathErrPanel, aiePathErrLabel, aiePathTextField, exception.getMessage());
            return false;
        }

        return true;
    }

    private boolean isModelFileValid() {
        String modelFile = modelFileTextField.getText();
        if (StringUtils.isEmpty(modelFile)) {
            modelFileTextField.setBorder(BorderFactory.createLineBorder(JBColor.RED));
            abnormalInput(modelFileErrPanel, modelFileErrLabel, modelFileTextField,
                    INPUT_MODEL_FILE_IS_EMPTY);
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
                    OUTPUT_MODEL_NAME_IS_EMPTY);
            return false;
        }

        if (!modelName.matches(FILE_NAME_REGULAR_EXPRESSION_PATTERN)) {
            abnormalInput(modelNameErrPanel, modelNameErrLabel, modelNameTextField, VALID_FILE_NAME_CHARACTERS);

            return false;
        }

        if (modelName.length() > MODEL_NAME_LIMIT) {
            abnormalInput(modelNameErrPanel, modelNameErrLabel, modelNameTextField, MODEL_NAME_MAXIMUM);
            return false;
        }

        return true;
    }

    private boolean isOutputPathValid() {
        String outputPath = outputPathTextField.getText();
        if (StringUtils.isEmpty(outputPath)) {
            outputPathTextField.setBorder(BorderFactory.createLineBorder(JBColor.RED));
            abnormalInput(outputPathErrPanel, outputPathErrLabel, outputPathTextField,
                    OUTPUT_PATH_IS_EMPTY);
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

        isValid &= isCannPathValid();
        isValid &= isAiePathValid();
        isValid &= isModelFileValid();
        isValid &= checkModelNameValid();
        isValid &= isOutputPathValid();

        return isValid;
    }

    @Override
    protected void doOKAction() {
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

        String outputFile = FileUtils.getFile(outputPathTextField.getText(),
                                                modelNameTextField.getText() + OM_MODEL_FILE_SUFFIX).toString();
        if (checkFileExist(outputFile)) {
            int ret = Messages.showYesNoDialog(REPLACE_THE_FOLLOWING_MODEL_OR_USE_ANOTHER_NAME
                                            + System.getProperty("line.separator") + outputPathTextField.getText(),
                                        "Info", "Replace", "Rename",null);
            if (ret != Messages.YES) {
                return;
            }
        }

        ProgressManager.getInstance().run(new AitModelConvertTask(project, cannPathTextField.getText(),
                                aiePathTextField.getText(), modelFileTextField.getText(),
                                Objects.requireNonNull(socVersionComboBox.getSelectedItem()).toString(), outputFile));
        super.doOKAction();
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
