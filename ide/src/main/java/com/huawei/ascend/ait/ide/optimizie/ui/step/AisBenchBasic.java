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

import com.huawei.ascend.ait.ide.commonlib.util.safeCmd.CmdStrBuffer;
import com.huawei.ascend.ait.ide.util.FileChooseWithBrows;
import com.huawei.ascend.ait.ide.commonlib.ui.SwitchButton;

import com.intellij.openapi.fileChooser.FileChooserDescriptor;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.ui.DialogWrapper;
import com.intellij.openapi.ui.TextFieldWithBrowseButton;
import com.intellij.openapi.util.Comparing;

import org.jetbrains.annotations.Nullable;

import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.JToggleButton;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * AisBenchBasic
 *
 * @author cabbage
 * @date 2023/06/03
 */
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
        setIcons();

        setFileChoodeAction();
        initComponent();
        initVisible();
        setOKButtonText("Start");
    }

    private void initComponent() {
    }

    private void initVisible() {
    }

    private void setIcons() {
    }

    private void setFileChoodeAction() {
    }

    private void modelFIleAction() {
    }

    private void inputAction() {
    }

    private void outputAction() {
    }

    private void outputdIRAction() {
    }

    private void checkFileSize(File file) {
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
        close(0);
    }

    private CmdStrBuffer getCmdStrBuffer() {
        return null;
    }

    private Boolean preCheck() {
        return true;
    }

}

