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

package com.huawei.ascend.ait.ide.commonlib.plugintool;

/**
 * PluginClassId
 *
 * @author cabbage
 * @since 2023/06/03
 */
public class PluginClassId {
    /**
     * Ascend_id
     */
    public static final String ASCEND_ID = "com.huawei.mindstudio.ascend";

    /**
     * Foundation_PluginId
     */
    public static final String FOUNDATION_PLUGIN_ID = ASCEND_ID + ".foundation";

    /**
     * Profiler_PluginId
     */
    public static final String PROFILER_PLUGIN_ID = ASCEND_ID + ".profiler";

    /**
     * Inference_PluginId
     */
    public static final String INFERENCE_PLUGIN_ID = ASCEND_ID + ".inference";

    /**
     * AitIde_PluginId
     */
    public static final String AIT_IDE_PLUGIN_ID = "com.huawei.ascend.ait.ide";

    /**
     * ModelAnalyse_ClassId
     */
    public static final String MODEL_ANALYSE_CLASS_ID = ASCEND_ID + ".infer.action.ModelAnalyseAction";

    /**
     * ModelConverter_ClassId
     */
    public static final String AIT_MODEL_CONVERTER_CLASS_ID = "com.huawei.ascend.ait.ide.action.AitModelConverterAction";

    /**
     * SystemProfiler_ClassId
     */
    public static final String SYSTEM_PROFILER_CLASS_ID =
            ASCEND_ID + ".systemprofiling.actions.SystemProfilingProfileAction";

    /**
     * AisBench_ClassId
     */
    public static final String AIS_BENCH_CLASS_ID = "com.huawei.ascend.ait.ide.action.AisBenchAction";

    /**
     * Compare_ClassId
     */
    public static final String COMPARE_CLASS_ID = "com.huawei.ascend.ait.ide.action.CompareAction";
}
