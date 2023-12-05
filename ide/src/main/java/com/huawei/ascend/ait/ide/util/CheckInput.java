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

package com.huawei.ascend.ait.ide.util;

import static com.huawei.ascend.ait.ide.optimizie.ui.step.AitModelConverterStep.MODEL_FILE_NOT_EXIST;
import static com.huawei.ascend.ait.ide.optimizie.ui.step.AitModelConverterStep.NO_READ_PERMISSION;
import static com.huawei.ascend.ait.ide.optimizie.ui.step.AitModelConverterStep.PATH_MUST_BE_A_DIRECTORY;
import static com.huawei.ascend.ait.ide.optimizie.ui.step.AitModelConverterStep.PATH_NOT_EXIST;
import static com.huawei.ascend.ait.ide.optimizie.ui.step.AitModelConverterStep.PATH_REGULAR_EXPRESSION_PATTERN;
import static com.huawei.ascend.ait.ide.optimizie.ui.step.AitModelConverterStep.SOFT_LINK_PATH;
import static com.huawei.ascend.ait.ide.optimizie.ui.step.AitModelConverterStep.UNSUPPORTED_MODEL_TYPE;

import com.huawei.ascend.ait.ide.util.exception.ModelFileInvalidException;
import com.huawei.ascend.ait.ide.util.exception.PathInvalidException;

import com.intellij.icons.AllIcons;
import com.intellij.openapi.ui.Messages;
import com.intellij.openapi.util.io.FileUtil;
import com.intellij.ui.JBColor;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.StringUtils;
import org.jetbrains.annotations.NotNull;

import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

/**
 * CheckInput
 *
 * @author cabbage
 * @since 2023/06/08
 */
public class CheckInput {
    private static final String DIGIT_EXPRESSION_PATTERN = "[0-9]+";
    private static final String STRING_EXPRESSION_PATTERN = "(,|[0-9])+";
    public static final int DOCUMENT_LIMIT = 256;
    public static final int STRING_LIMIT = 64;
    public static final String VALID_DIR_PATH_CHARACTERS = "Valid folder path characters: -, _, :, \\, /, [0-9], [A-Z], [a-z].";
    public static final String VALID_CHARTERS = "Only letters, digits, and the following special characters are allowed:- . _ : \\ /";
    public static final String VALID_STRING_CHARATERS = "Only digits, and the following special characters are allowed: , ";
    public static final String VALID_DIGITS_CHARATERS = "Only digits are allowed.";
    public static final String INPUT_MAXIMUM = "The maximum input size is 64 bits.";
    private static final long FILE_SIZE_MAX = (long) 2 * 1024 * 1024 * 1024;

    /**
     * checkFileExist
     *
     * @param path path
     * @return boolean
     */
    private static boolean checkFileExist(@NotNull String path) {
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

    public static int checkFileSize(File file) {
        if (file.length() > FILE_SIZE_MAX) {
            return Messages.showDialog("The size of the file: " + file + " exceed 2 GB, Do you want to continue?",
                    "Please Confirm", new String[]{"Yes", "No"},
                    Messages.NO, AllIcons.General.QuestionDialog);
        }
        return Messages.YES;
    }

    private static boolean pathValid(@NotNull String path) {
        return path.matches(PATH_REGULAR_EXPRESSION_PATTERN);
    }

    /**
     * checkPath
     *
     * @param jTextField jTextField
     * @param jLabel     jLabel
     * @param jPanel     jPanel
     */
    public static void checkPath(JTextField jTextField, JLabel jLabel, JPanel jPanel) {
        String path= jTextField.getText();
        if (StringUtils.isEmpty(path)) {
            normalizeInput(jPanel, jLabel, jTextField);
            return;
        }
        try {
            checkPathValid(path);
            normalizeInput(jPanel, jLabel, jTextField);
        } catch (PathInvalidException e) {
            abnormalInput(jPanel, jLabel, jTextField, e.getMessage());
        }
    }

    /**
     * checkPathValid
     *
     * @param path path
     * @throws PathInvalidException PathInvalidException
     */
    private static void checkPathValid(@NotNull String path) throws PathInvalidException {
        if (!path.matches(PATH_REGULAR_EXPRESSION_PATTERN)) {
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

    /**
     * FileCheck
     *
     * @param modelErrorJPanel   modelErrorJPanel
     * @param modelErrorJLabel   modelErrorJLabel
     * @param modelFileTextField modelFileTextField
     * @param fileSuffix     fileSuffix
     */
    public static void FileCheck(JPanel modelErrorJPanel, JLabel modelErrorJLabel, JTextField modelFileTextField, List<String> fileSuffix) {
        String modelFile = modelFileTextField.getText();
        if (StringUtils.isEmpty(modelFile)) {
            normalizeInput(modelErrorJPanel, modelErrorJLabel, modelFileTextField);
            return;
        }

        try {
            checkModelFileValid(modelFile, fileSuffix);
            normalizeInput(modelErrorJPanel, modelErrorJLabel, modelFileTextField);
        } catch (ModelFileInvalidException exception) {
            abnormalInput(modelErrorJPanel, modelErrorJLabel, modelFileTextField, exception.getMessage());
        }
    }

    public static void checkModelFileValid(@NotNull String modelFile, List<String> fileSuffix) throws ModelFileInvalidException {
        if (!pathValid(modelFile)) {
            throw new ModelFileInvalidException(VALID_DIR_PATH_CHARACTERS);
        }
        if (!isSupportFileType(modelFile, fileSuffix)) {
            throw new ModelFileInvalidException(UNSUPPORTED_MODEL_TYPE);
        }
        if (!checkFileExist(modelFile)) {
            throw new ModelFileInvalidException(MODEL_FILE_NOT_EXIST);
        }
        if (!FileUtils.getFile(modelFile).canRead()) {
            throw new ModelFileInvalidException(NO_READ_PERMISSION);
        }
    }

    private static boolean isSupportFileType(String modelFile, List<String> fileSuffix) {
        for (String suffix : fileSuffix) {
            if (modelFile.endsWith(suffix)) {
                return true;
            }
        }
        return false;
    }

    /**
     * checkStringValid
     *
     * @param errorJPanel errorJPanel
     * @param errorJLabel errorJLabel
     * @param jText       jText
     * @return boolean
     */
    public static boolean checkStringValid(JPanel errorJPanel, JLabel errorJLabel, JTextField jText) {
        String text = jText.getText();
        if (!text.matches(STRING_EXPRESSION_PATTERN)) {
            abnormalInput(errorJPanel, errorJLabel, jText, VALID_STRING_CHARATERS);
            return false;
        }

        if (text.length() > STRING_LIMIT) {
            abnormalInput(errorJPanel, errorJLabel, jText, INPUT_MAXIMUM);
            return false;
        }
        return true;
    }

    /**
     * checkDigitValid
     *
     * @param errorJPanel errorJPanel
     * @param errorJLabel errorJLabel
     * @param jText       jText
     * @return boolean
     */
    public static boolean checkDigitValid(JPanel errorJPanel, JLabel errorJLabel, JTextField jText) {
        String text = jText.getText();
        if (!text.matches(DIGIT_EXPRESSION_PATTERN)) {
            abnormalInput(errorJPanel, errorJLabel, jText, VALID_DIGITS_CHARATERS);
            return false;
        }

        if (text.length() > STRING_LIMIT) {
            abnormalInput(errorJPanel, errorJLabel, jText, INPUT_MAXIMUM);
            return false;
        }
        return true;
    }

    /**
     * normalizeInput
     *
     * @param panel  panel
     * @param label label
     * @param textField textField
     */
    public static void normalizeInput(@NotNull JPanel panel, @NotNull JLabel label,
                                      @NotNull JTextField textField) {
        label.setText("");
        panel.setVisible(false);
    }

    private static void abnormalInput(@NotNull JPanel errPanel, @NotNull JLabel errLabel,
                                      @NotNull JTextField textField, @NotNull String errMsg) {
        errPanel.setVisible(true);
        errLabel.setText(errMsg);
        errLabel.setForeground(JBColor.RED);
    }
}


