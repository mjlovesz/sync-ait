package com.ascend.ait.ide.optimizie.ui.step;

import com.ascend.ait.ide.Icons;
import com.ascend.ait.ide.commlib.ui.SwitchButton;
import com.ascend.ait.ide.util.FileChooseWithBrows;
import com.ascend.ait.ide.commlib.exception.CommandInjectException;
import com.ascend.ait.ide.commlib.output.OutputService;
import com.ascend.ait.ide.commlib.util.safeCmd.CmdExec;
import com.ascend.ait.ide.commlib.util.safeCmd.CmdStrBuffer;
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
import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class ais_bench_basic extends DialogWrapper {
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
    private final JTextField modelFileTextField = modelFileBrowse.getTextField();
    private final JTextField inputFilesTextField = inputFileBrowse.getTextField();
    private final JTextField outputTextField = outputPathBrowse.getTextField();
    private final JTextField outputDirTextField = outputDirnameBrowse.getTextField();

    private static final String OM_MODEL_FILE_EXTENSION = "java";
    private static final String NPY_FILE_EXTENSION = "npy";
    private static final String BIN_FILE_EXTENSION = "bin";
    private static final String TXT_FILE_EXTENSION = "txt";
    private static final List<String> PURE_DATA_TYPE = List.of("zero", "random");
    private static final List<String> OUTFMT_TYPE = List.of("BIN", "NPY", "TXT");
    private static final long FILE_SIZE_MAX = (long) 2 * 1024 * 1024 * 1024;
    private final Project project;
    private JComponent debugButton;
    private SwitchButton displayButton;

    public ais_bench_basic(Project project) {
        super(true);
        this.project = project;
        init();
        setIcons();

        setFileChoodeAction();
        initComponent();
        initVisible();
        setOKButtonText("Start");
    }

    private void initVisible() {
        advance.setVisible(false);

        pureDataTypeJLabel.setVisible(false);
        pureDataTypeCombx.setVisible(false);

        outputDirnameJLabel.setVisible(false);
        outputDirnameBrowse.setVisible(false);

        outFmtJLabel.setVisible(false);
        outFormatComboBox.setVisible(false);
    }

    private void setIcons() {
        model.setIcon(Icons.STAR);
    }

    private void initComponent() {
        for (String s : PURE_DATA_TYPE) {
            this.pureDataTypeCombx.addItem(s);
        }
        for (String s : OUTFMT_TYPE) {
            this.outFormatComboBox.addItem(s);
        }
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
                .withTitle("model file")
                .withDescription("teste");
        return FileChooseWithBrows.fileChoosewithBrowse(project, fileChooserDescriptor,
                this.getClass().getName(), "modelseletcPath").orElse(null);
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
        Messages.showErrorDialog("Check", "ERROR");
        OutputService.getInstance(project).print("testeeee");
        OutputService.getInstance(project).print(modelFileTextField.getText());
        OutputService.getInstance(project).print(loopTextField.getText());

        CmdStrBuffer cmdStrBuffer = new CmdStrBuffer();
        cmdStrBuffer.append("dir");
        OutputService.getInstance(project).print(cmdStrBuffer.toString());
        CmdExec exec = new CmdExec();
        try {
            exec.bashStart(cmdStrBuffer);
            OutputService.getInstance(project).print("TEST", ConsoleViewContentType.LOG_INFO_OUTPUT);
            String execRec = exec.getResult();
            if (execRec != null) {
                OutputService.getInstance(project).print(execRec, ConsoleViewContentType.LOG_DEBUG_OUTPUT);
            }
            OutputService.getInstance(project).print("error", ConsoleViewContentType.LOG_ERROR_OUTPUT);
        } catch (CommandInjectException | IOException e) {
            throw new RuntimeException(e);
        }
        close(0);
    }

    /*
    在下发cmd之前检查配置是否完善
    1：是否配置model file
    2：有依赖关系的配置是否完成配置。
    check weather the input is correct
     */
    private Boolean preCheck() {
        String modelfile = modelFileTextField.getText();
        if (modelfile.isEmpty()) {
            Messages.showErrorDialog("Model file must be chose", "ERROR");
            return false;
        }
        return true;
    }

    /*
    检查对于device loop warmup三个输入的文本是否正确
   */
    private Boolean inputValidCheck() {
        return true;
    }

}

