/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

package com.ascend.ait.ide.commlib.util;

import org.jetbrains.annotations.NotNull;

import java.util.Locale;
import java.util.ResourceBundle;

/**
 * get strings through resource bundle
 *
 * @author Shengshuai Huang
 * @since 2019/8/15 17:11
 */

public class BundleUtil {
    /**
     * Get bundle string by resource bundle and key
     *
     * @param bundle the bundle
     * @param key    the key
     * @return the value by the key and the resource bundle
     */
    public static String getString(@NotNull Bundles bundle, String key) {
        ResourceBundle resourceBundle = ResourceBundle.getBundle(bundle.getBundleName(), Locale.ENGLISH);
        return resourceBundle.getString(key);
    }

    /**
     * Get profiling bundles value by key.
     *
     * @param key the key
     * @return the value by the key
     */
    public static String getProfilingString(String key) {
        return getString(Bundles.PROFILING, key);
    }

    /**
     * Get common libs bundles value by key.
     *
     * @param key the key
     * @return the value by the key
     */
    public static String getCommonlibsString(String key) {
        return getString(Bundles.COMMONLIB, key);
    }

    /**
     * Get environment variables bundles value by key.
     *
     * @param key the key
     * @return the value by the key
     */
    public static String getEnvVarsString(String key) {
        return getString(Bundles.ENVIRONMENT, key);
    }

    enum Bundles {
        /**
         * ResourceBundle name for profiling module
         */
        PROFILING("i18n/profilingCommonLibs"),

        /**
         * ResourceBundle name for common libs
         */
        COMMONLIB("i18n.commonlibs"),

        /**
         * ResourceBundle name for environment variables
         */
        ENVIRONMENT("i18n.environmentVars");

        private String bundleName;

        Bundles(String bundleName) {
            this.bundleName = bundleName;
        }

        public String getBundleName() {
            return bundleName;
        }

        @Override
        public String toString() {
            return bundleName;
        }
    }
}
