package com.ascend.ait.ide.commonlib.icons;

import com.ascend.ait.ide.Icons;
import com.intellij.openapi.util.IconLoader;

import javax.swing.Icon;

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

    public static final Icon SWITCH_CLOSE = IconLoader.findIcon(
            "/icons/switchclose.svg", Icons.class.getClassLoader());

    public static final Icon SWITCH_OPEN = IconLoader.findIcon(
            "/icons/switchopen.svg", Icons.class.getClassLoader());
}
