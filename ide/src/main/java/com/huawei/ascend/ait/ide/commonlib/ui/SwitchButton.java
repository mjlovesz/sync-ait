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

package com.huawei.ascend.ait.ide.commonlib.ui;

import com.huawei.ascend.ait.ide.commonlib.icons.CommonLibIcons;

import javax.swing.Icon;
import javax.swing.JLabel;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

/**
 * switch button
 */
public class SwitchButton extends JLabel {
    private final Icon onIcon = CommonLibIcons.SWITCH_OPEN;
    private final Icon offIcon = CommonLibIcons.SWITCH_CLOSE;
    private boolean isSelected;
    private StateChangeListener listener;
    private boolean isSwitchButtonEnable;

    private final MouseAdapter mouseAdapter = new MouseAdapter() {
        @Override
        public void mouseClicked(MouseEvent mouseEvent) {
            super.mouseClicked(mouseEvent);
            setSwitchStatus(!isSelected);
        }
    };

    /**
     * SwitchButton
     */
    public SwitchButton() {
        this(false);
    }

    public SwitchButton(boolean isOn) {
        isSwitchButtonEnable = true;
        setSwitchStatus(isOn);
        addMouseListener(mouseAdapter);
    }

    /**
     * set switch button enable
     *
     * @param isEnable Enable flag
     */
    public void setEnable(boolean isEnable) {
        if (isEnable == isSwitchButtonEnable) {
            return;
        }

        isSwitchButtonEnable = isEnable;
        if (isEnable) {
            addMouseListener(mouseAdapter);
        } else {
            removeMouseListener(mouseAdapter);
        }
    }

    /**
     * is Switch Button Enable
     *
     * @return boolean
     */
    public boolean isSwitchButtonEnable() {
        return isSwitchButtonEnable;
    }

    /**
     * set On
     */
    public void setOn() {
        setSwitchStatus(true);
    }

    /**
     * set Off
     */
    public void setOff() {
        setSwitchStatus(false);
    }

    /**
     * turn on switch
     *
     * @param isSelected isSelected
     */
    public void setSwitchStatus(boolean isSelected) {
        this.isSelected = isSelected;
        setIcon(this.isSelected ? onIcon : offIcon);
        if (listener != null) {
            listener.onChange(isSelected);
        }
    }

    /**
     * is select button
     *
     * @return boolean
     */
    public boolean isSelected() {
        return isSelected;
    }

    /**
     * add listener
     *
     * @param listener listener
     */
    public void addListener(StateChangeListener listener) {
        this.listener = listener;
    }

    /**
     * state change listener
     */
    public interface StateChangeListener {
        /**
         * state change callback
         *
         * @param isSelected isSelected
         */
        void onChange(boolean isSelected);
    }
}

