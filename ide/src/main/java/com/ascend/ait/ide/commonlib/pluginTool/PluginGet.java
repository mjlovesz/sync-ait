package com.ascend.ait.ide.commonlib.pluginTool;

import com.intellij.ide.plugins.PluginManager;
import com.intellij.openapi.extensions.PluginId;
import com.intellij.openapi.project.Project;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

public class PluginGet {
    private final Project project;
    public PluginGet(Project project) {
        this.project = project;
    }

    public void getPluginClass(String className, String id) {
        PluginId pluginId = PluginId.getId(id);
        try {
            Class a = PluginManager.getInstance().findEnabledPlugin(pluginId).getPluginClassLoader().loadClass(className);
            Method method = a.getMethod("openNewPage", void.class);
            Constructor constructor = a.getConstructor();
            Object object = constructor.newInstance();
            method.invoke(object, project);

        } catch (ClassNotFoundException | NoSuchMethodException | InvocationTargetException |
                 IllegalAccessException | InstantiationException e) {
            throw new RuntimeException(e);
        }
    }
}
