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

package com.huawei.ascend.ait.ide.filetype;

import com.huawei.ascend.ait.ide.optimizie.ui.Choosedialog;

import com.intellij.openapi.fileEditor.FileEditor;
import com.intellij.openapi.fileEditor.FileEditorState;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.util.UserDataHolderBase;
import com.intellij.openapi.vfs.VirtualFile;
import com.intellij.testFramework.LightVirtualFile;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import javax.swing.JComponent;
import java.beans.PropertyChangeListener;

/**
 * AitFileEditor
 *
 * @author cabbage
 * @since 2023/06/03
 */
public class AitFileEditor extends UserDataHolderBase implements FileEditor {
    private final Choosedialog choosedialog;
    private VirtualFile file;
    private Project project;

    /**
     * AitFileEditor
     *
     * @param project     project
     * @param virtualFile virtualFile
     */
    public AitFileEditor(@NotNull Project project, @NotNull VirtualFile virtualFile) {
        this.project = project;
        if (file instanceof LightVirtualFile) {
            LightVirtualFile lightVirtualFile = (LightVirtualFile) virtualFile;
            this.file = lightVirtualFile.getOriginalFile();
        }
        choosedialog = new Choosedialog(project);
        this.file = virtualFile;
    }

    @Override
    public @NotNull JComponent getComponent() {
        return choosedialog.getRoot();
    }

    @Override
    public @Nullable JComponent getPreferredFocusedComponent() {
        return null;
    }

    @Override
    public @NotNull String getName() {
        return AitFileEditorProvider.FILE_NAME;
    }

    @Override
    public void setState(@NotNull FileEditorState state) {
    }

    @Override
    public boolean isModified() {
        return false;
    }

    @Override
    public boolean isValid() {
        return true;
    }

    @Override
    public void addPropertyChangeListener(@NotNull PropertyChangeListener listener) {
    }

    @Override
    public void removePropertyChangeListener(@NotNull PropertyChangeListener listener) {
    }

    @Override
    public void dispose() {
    }

    @Nullable
    public VirtualFile getFile() {
        return this.file;
    }
}
