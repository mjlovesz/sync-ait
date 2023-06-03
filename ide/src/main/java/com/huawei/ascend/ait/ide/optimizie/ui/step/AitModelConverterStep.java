package com.huawei.ascend.ait.ide.optimizie.ui.step;

import com.huawei.ascend.ait.ide.optimizie.aitmodelconvert.AitModelConvertTask;
import com.huawei.ascend.ait.ide.util.FileChooseWithBrows;
import com.huawei.ascend.ait.ide.util.exception.OutputPathInvalidException;
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

import javax.swing.*;
import javax.swing.border.Border;
import javax.swing.event.DocumentEvent;
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
    private final Project project;
    private static final List<String> SOC_VERSION_LIST = List.of("Ascend310", "Ascend310P3");
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
        addModelFileBrowseListener();
        addoutputPathBrowseListener();
        addModelFileTextFieldListener();
        addModelNameTextListener();
        initOutputPathTextListener();
    }

    private void setTextFieldLimitAndToolTip() {
        modelFileTextField.setDocument(new JTextFieldLimit(DOCUMENT_LIMIT));
        modelFileTextField.setToolTipText("Only letters, digits, and the following special characters are allowed:- . _ : \\ /");
        modelNameTextField.setDocument(new JTextFieldLimit(DOCUMENT_LIMIT));
        modelNameTextField.setToolTipText("Only letters, digits, hyphens(-), end(.) and underscores(_) are allowed.");
        outputPathTextField.setDocument(new JTextFieldLimit(DOCUMENT_LIMIT));
        outputPathTextField.setToolTipText("Only letters, digits, and the following special characters are allowed:- . _ : \\ /");
    }

    private void initComponent() {
        for (String soc : SOC_VERSION_LIST) {
            this.socVersionComboBox.addItem(soc);
        }
        this.socVersionComboBox.setSelectedItem(SOC_VERSION_LIST.get(0));
    }

    private void initVisible() {
        socVersionComboBox.setVisible(true);
        modelFileBrowse.setVisible(true);
        outputPathBrowse.setVisible(true);
        modelFileErrPanel.setVisible(false);
        outputPathErrPanel.setVisible(false);
        modelNameTextField.setVisible(true);
        modelNameErrPanel.setVisible(false);
    }

    @Override
    protected @Nullable JComponent createCenterPanel() {
        return root;
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
                } catch (OutputPathInvalidException exception) {
                    abnormalInput(outputPathErrPanel, outputPathErrLabel, outputPathTextField, exception.getMessage());
                }
            }
        });
    }

    private void checkOutputPathValid(@NotNull String outputPath) throws OutputPathInvalidException {
        File outputPathFile = FileUtils.getFile(outputPath);
        if (!pathValid(outputPath)) {
            throw new OutputPathInvalidException("Valid file path characters: -, _, :, \\, /, [0-9], [A-Z], [a-z].");
        }
        if (FileUtils.isSymlink(outputPathFile)) {
            throw new OutputPathInvalidException("The path is a soft link.");
        }
        if (!FileUtil.exists(outputPath)) {
            throw new OutputPathInvalidException("The path does not exist.");
        }
        if (!FileUtils.getFile(outputPath).isDirectory()) {
            throw new OutputPathInvalidException("The path must be a directory.");
        }
        if (!FileUtils.getFile(outputPath).canWrite()) {
            throw new OutputPathInvalidException("You do not have the permission to create files in the current directory.");
        }
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
        if (OS.isLinux() && (!isOtherWritableFile(modelFile))) {
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

    private void addoutputPathBrowseListener() {
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
        } catch (OutputPathInvalidException exception) {
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

        ProgressManager.getInstance().run(new AitModelConvertTask(project, modelFileTextField.getText(),
                Objects.requireNonNull(socVersionComboBox.getSelectedItem()).toString(), outputFile));
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
