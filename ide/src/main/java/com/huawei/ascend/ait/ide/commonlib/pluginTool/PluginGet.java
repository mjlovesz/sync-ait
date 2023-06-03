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

package com.huawei.ascend.ait.ide.commonlib.pluginTool;

import com.intellij.ide.plugins.PluginManager;
import com.intellij.openapi.extensions.PluginId;
import com.intellij.openapi.project.Project;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

/**
 * PluginGet
 *
 * @author cabbage
 * @date 2023/06/03
 */
public class PluginGet {
    private final Project project;
    public PluginGet(Project project) {
        this.project = project;
    }

    /**
     * getPluginClass
     *
     * @param className className
     * @param id id
     */
    public void getPluginClass(String className, String id) {
        PluginId pluginId = PluginId.getId(id);
        try {
            Class a = PluginManager.getInstance().findEnabledPlugin(pluginId).getPluginClassLoader().loadClass(className);
            Method method = a.getMethod("openNewPage", Project.class);
            Constructor constructor = a.getConstructor();
            Object object = constructor.newInstance();
            method.invoke(object, project);
        } catch (ClassNotFoundException | NoSuchMethodException | InvocationTargetException |
                 IllegalAccessException | InstantiationException e) {
            throw new RuntimeException(e);
        }
    }
}
