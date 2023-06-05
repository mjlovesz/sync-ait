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

import com.intellij.openapi.fileChooser.FileChooser;
import com.intellij.openapi.fileChooser.FileChooserDescriptor;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.vfs.LocalFileSystem;
import com.intellij.openapi.vfs.VirtualFile;
import com.intellij.openapi.vfs.VirtualFileManager;
import org.apache.commons.lang.StringUtils;
import org.jetbrains.annotations.Nullable;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.prefs.Preferences;

public class FileChooseWithBrows {
    public static Optional<String> fileChoosewithBrowse(Project project,
                                                        FileChooserDescriptor descriptor,
                                                        String className, @Nullable String historyKey) {
        return fileChoosewithBrowse(project, descriptor, new ValueSaveInterFace() {

            @Override
            public void set(String val) {
                String classNameWithOutPackage = className;
                if (className.lastIndexOf(".") != -1) {
                    classNameWithOutPackage = className.substring(className.lastIndexOf("."));
                }
                Preferences.userRoot().node(project.getName() + classNameWithOutPackage).put(historyKey, val);
            }

            @Override
            public String get(String valDefault) {
                String classNameWithOutPackage = className;
                if (className.lastIndexOf(".") != -1) {
                    classNameWithOutPackage = className.substring(className.lastIndexOf("."));
                }
                return Preferences.userRoot().node(project.getName() + classNameWithOutPackage)
                        .get(historyKey, valDefault);
            }
        });
    }

    public static Optional<String> fileChoosewithBrowse(Project project, FileChooserDescriptor descriptor,
                                                        ValueSaveInterFace valueSaveInterFace) {
        String lastStorePath = valueSaveInterFace.get(System.getProperty("user.home"));
        String[] lastFilePath = StringUtils.split(lastStorePath, ',');
        String lastPath = null;
        if (lastFilePath != null && lastFilePath.length > 0) {
            lastPath = lastFilePath[0];
        }
        if (lastPath == null) {
            lastPath = System.getProperty("user.home");
        }

        VirtualFileManager.getInstance().refreshWithoutFileWatcher(true);
        VirtualFile lastSelectFile = LocalFileSystem.getInstance().findFileByPath(lastPath);
        VirtualFile[] virtualFiles = FileChooser.chooseFiles(descriptor, project, lastSelectFile);

        if (virtualFiles == null || virtualFiles.length == 0) {
            return Optional.empty();
        }

        List<String> filePaths = new ArrayList<>();
        for (VirtualFile virtualFile : virtualFiles) {
            filePaths.add(virtualFile.getPath());
        }

        String selectedFilePath = StringUtils.join(filePaths, ",");

        if (selectedFilePath == null || selectedFilePath.isEmpty()) {
            return Optional.empty();
        }
        valueSaveInterFace.set(selectedFilePath);
        return Optional.of(selectedFilePath);
    }

    public interface ValueSaveInterFace {
        void set(String val);
        String get(String valDefault);
    }
}
