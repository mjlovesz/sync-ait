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

import com.intellij.openapi.fileChooser.FileChooserDescriptor;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.vfs.VirtualFile;

import java.util.ArrayList;
import java.util.List;

/**
 * FileChoose
 *
 * @author cabbage
 * @since 2023/06/05
 */
public class FileChoose {

    /**
     * getSelectedFile
     *
     */
    public static String getSelectedFile(Project project, List<String> strings, Boolean chooseMultiple, String historyKey) {
        FileChooserDescriptor fileChooserDescriptor = new FileChooserDescriptor(true, false, false, false,
                false, chooseMultiple)
                .withFileFilter(virtualFile -> virtualFile.isDirectory() || new ArrayList<>(strings).contains(virtualFile.getExtension()))
                .withTitle("Browse for File")
                .withDescription("Please select the appropriate file of " + strings);
        return FileChooseWithBrows.fileChoosewithBrowse(project, fileChooserDescriptor,
                "", historyKey).orElse(null);
    }

    /**
     * getSelectedPath
     *
     */
    public static String getSelectedPath(Project project, String historyKey) {
        FileChooserDescriptor fileChooserDescriptor = new FileChooserDescriptor(false, true, false, false,
                false, false)
                .withFileFilter(VirtualFile::isDirectory)
                .withTitle("Browse for Path")
                .withDescription("Select the appropriate path");
        return FileChooseWithBrows.fileChoosewithBrowse(project, fileChooserDescriptor,
                "", historyKey).orElse(null);
    }
}
