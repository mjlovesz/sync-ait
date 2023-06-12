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

package com.huawei.ascend.ait.ide.service;

import com.intellij.ide.ApplicationLoadListener;
import com.intellij.ide.plugins.IdeaPluginDescriptorImpl;
import com.intellij.ide.plugins.PluginManagerCore;
import com.intellij.ide.plugins.RawPluginDescriptor;
import com.intellij.openapi.application.Application;
import org.jetbrains.annotations.NotNull;

import java.nio.file.Path;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;

public class AscendMenuAction implements ApplicationLoadListener {

    @Override
    public void beforeApplicationLoaded(@NotNull Application application, @NotNull Path configPath) {
        excludeDuplicateAction();
    }

    private void excludeDuplicateAction() {
        List<IdeaPluginDescriptorImpl> modules = PluginManagerCore.getPluginSet().getEnabledModules();
        HashSet<String> idSet = new HashSet<>();

        for (IdeaPluginDescriptorImpl model : modules) {
            Iterator<RawPluginDescriptor.ActionDescriptor> actions = model.actions.iterator();
            while (actions.hasNext()) {
                RawPluginDescriptor.ActionDescriptor next = actions.next();
                String id = next.element.attributes.get("id");
                if (id == null || !id.startsWith("MindStudio")) {
                    continue;
                }
                if (idSet.contains(id)) {
                    actions.remove();
                } else {
                    idSet.add(id);
                }
            }
        }
    }
}
