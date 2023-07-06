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

/**
 * PluginClassId
 *
 * @author cabbage
 * @date 2023/06/03
 */
public class PluginClassId {
    /**
     * Ascend_id
     */
    public static final String Ascend_id = "com.huawei.mindstudio.ascend";

    /**
     * Foundation_PluginId
     */
    public static final String Foundation_PluginId = Ascend_id + ".foundation";

    /**
     * Profiler_PluginId
     */
    public static final String Profiler_PluginId = Ascend_id + ".profiler";

    /**
     * Inference_PluginId
     */
    public static final String Inference_PluginId = Ascend_id + ".inference";

    /**
     * AitIde_PluginId
     */
    public static final String AitIde_PluginId = "com.huawei.ascend.ait.ide";

    /**
     * ModelAnalyse_ClassId
     */
    public static final String ModelAnalyse_ClassId = Ascend_id + ".infer.action.ModelAnalyseAction";

    /**
     * ModelConverter_ClassId
     */
    public static final String AitModelConverter_ClassId = "com.huawei.ascend.ait.ide.action.AitModelConverterAction";

    /**
     * SystemProfiler_ClassId
     */
    public static final String SystemProfiler_ClassId = Ascend_id + ".systemprofiling.actions.SystemProfilingProfileAction";

    /**
     * AisBench_ClassId
     */
    public static final String AisBench_ClassId = "com.huawei.ascend.ait.ide.action.AisBenchAction";

    /**
     * Compare_ClassId
     */
    public static final String Compare_ClassId = "com.huawei.ascend.ait.ide.action.CompareAction";
}
