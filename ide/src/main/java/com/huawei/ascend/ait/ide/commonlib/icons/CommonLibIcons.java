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

package com.huawei.ascend.ait.ide.commonlib.icons;

import com.huawei.ascend.ait.ide.Icons;
import com.intellij.openapi.util.IconLoader;

import javax.swing.Icon;

/**
 * CommonLib Icons
 *
 * @author cabbage
 * @since 2023/06/03
 */
public interface CommonLibIcons {
    /**
     * TOOL_ICON
     */
    Icon TOOL_ICON = IconLoader.getIcon("/icons/light/output.svg", CommonLibIcons.class);

    /**
     * TOOL_ICON_DARK
     */
    Icon TOOL_ICON_DARK = IconLoader.getIcon("/icons/dark/output.svg", CommonLibIcons.class);

    /**
     * DETAIL_ICON
     */
    Icon DETAIL_ICON = IconLoader.getIcon("/icons/light/detail.svg", CommonLibIcons.class);

    /**
     * DETAIL_ICON_DARK
     */
    Icon DETAIL_ICON_DARK = IconLoader.getIcon("/icons/dark/detail.svg", CommonLibIcons.class);

    /**
     * SWITCH_CLOSE
     */
    Icon SWITCH_CLOSE = IconLoader.findIcon(
            "/icons/switchclose.svg", Icons.class.getClassLoader());

    /**
     * SWITCH_OPEN
     */
    Icon SWITCH_OPEN = IconLoader.findIcon(
            "/icons/switchopen.svg", Icons.class.getClassLoader());
}
