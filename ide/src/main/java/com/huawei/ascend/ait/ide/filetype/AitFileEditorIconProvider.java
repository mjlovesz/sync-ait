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

import com.huawei.ascend.ait.ide.Icons;
import com.huawei.ascend.ait.ide.commonlib.ui.UiUtils;

import com.intellij.ide.FileIconProvider;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.vfs.VirtualFile;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import javax.swing.Icon;

/**
 * AitFileEditorIconProvider
 *
 * @author cabbage
 * @since 2023/06/08
 */
public class AitFileEditorIconProvider implements FileIconProvider {
    private static final String AIT_FILE = "AIT";

    @Override
    public @Nullable Icon getIcon(@NotNull VirtualFile file, int flags, @Nullable Project project) {
        if (AIT_FILE.equals(file.getName())) {
            return UiUtils.getJbIcon(Icons.AIT_TITLE_DARK, Icons.AIT_TITLE_LIGHT);
        }
        return null;
    }
}
