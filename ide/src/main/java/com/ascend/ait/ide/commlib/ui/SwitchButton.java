package com.ascend.ait.ide.commlib.ui;


import com.ascend.ait.ide.Icons;


import javax.swing.Icon;
import javax.swing.JLabel;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

/**
 * switch button
 */
public class SwitchButton extends JLabel {
    private final Icon onIcon = Icons.SWITCH_OPEN;
    private final Icon offIcon = Icons.SWITCH_CLOSE;
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

    public SwitchButton() {
        this(false);
    }

    public SwitchButton(boolean on) {
        isSwitchButtonEnable = true;
        setSwitchStatus(on);
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
     * @param selected selected
     */
    public void setSwitchStatus(boolean selected) {
        isSelected = selected;
        setIcon(isSelected ? onIcon : offIcon);
        if (listener != null) {
            listener.onChange(selected);
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
     *
     * @since 2019.10.14
     * @author baiguochao
     */
    public interface StateChangeListener {
        /**
         * state change callback
         *
         * @param selected selected
         */
        void onChange(boolean selected);
    }
}

