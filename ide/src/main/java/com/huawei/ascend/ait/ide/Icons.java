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

package com.huawei.ascend.ait.ide;

import com.huawei.ascend.ait.ide.action.AitAction;
import com.intellij.openapi.util.IconLoader;

import javax.swing.Icon;

/**
 * Icons
 *
 * @author cabbage
 * @since 2023/06/03
 */
public class Icons {
    public static final Icon AIS_BENCH_DARK = IconLoader.findIcon(
            "/icons/AisBench_dark.svg", Icons.class.getClassLoader());

    public static final Icon AIS_BENCH_LIGHT = IconLoader.findIcon(
            "/icons/AisBench_light.svg", Icons.class.getClassLoader());

    public static final Icon COMPARE_DARK = IconLoader.findIcon(
            "/icons/Compare_dark.svg", Icons.class.getClassLoader());

    public static final Icon COMPARE_LIGHT = IconLoader.findIcon(
            "/icons/Compare_light.svg", Icons.class.getClassLoader());

    public static final Icon RIGHT_DARK = IconLoader.findIcon(
            "/icons/dark@1x.png", Icons.class.getClassLoader());

    public static final Icon RIGHT_LIGHT = IconLoader.findIcon(
            "/icons/light@1x.png", Icons.class.getClassLoader());

    public static final Icon MODEL_ANALYSE_DARK = IconLoader.findIcon(
            "/icons/ModelAnalyse_dark.svg", Icons.class.getClassLoader());

    public static final Icon MODEL_ANALYSE_LIGHT = IconLoader.findIcon(
            "/icons/ModelAnalyse_light.svg", Icons.class.getClassLoader());

    public static final Icon AIT_MODEL_CONVERTER_DARK = IconLoader.findIcon(
            "/icons/ModelConverter_dark.svg", Icons.class.getClassLoader());

    public static final Icon AIT_MODEL_CONVERTER_LIGHT = IconLoader.findIcon(
            "/icons/ModelConverter_light.svg", Icons.class.getClassLoader());

    public static final Icon QUESTION = IconLoader.findIcon(
            "/icons/question-mark-new.png", Icons.class.getClassLoader());

    public static final Icon STAR = IconLoader.findIcon(
            "/icons/star.png", Icons.class.getClassLoader());

    public static final Icon SYSTEM_PROFILER_DARK = IconLoader.findIcon(
            "/icons/SystemProfiler_dark.svg", Icons.class.getClassLoader());

    public static final Icon SYSTEM_PROFILER_LIGHT = IconLoader.findIcon(
            "/icons/SystemProfiler_light.svg", Icons.class.getClassLoader());

    public static final Icon AIS_TITLE_DARK = IconLoader.findIcon(
            "/icons/dark/AisBench_dark.svg", Icons.class.getClassLoader());

    public static final Icon AIS_TITLE_LIGHT = IconLoader.findIcon(
            "/icons/light/AisBench_light.svg", AitAction.class.getClassLoader());

    public static final Icon COMPARE_TITLE_DARK = IconLoader.findIcon(
            "/icons/dark/Compare_dark.svg", Icons.class.getClassLoader());

    public static final Icon COMPARE_TITLE_LIGHT = IconLoader.findIcon(
            "/icons/light/Compare_light.svg", AitAction.class.getClassLoader());

    public static final Icon AIT_TITLE_DARK = IconLoader.findIcon(
            "/icons/AIT_dark.svg", Icons.class.getClassLoader());

    public static final Icon AIT_TITLE_LIGHT = IconLoader.findIcon(
            "/icons/AIT_light.svg", AitAction.class.getClassLoader());

    public static final Icon MODEL_CONVERT_TITLE_DARK = IconLoader.findIcon(
            "/icons/dark/ModelConverter_dark.svg", Icons.class.getClassLoader());

    public static final Icon MODEL_CONVERT_TITLE_LIGHT = IconLoader.findIcon(
            "/icons/light/ModelAnalyse_light.svg", AitAction.class.getClassLoader());
}
