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

import com.intellij.ui.JBColor;
import com.intellij.util.ui.UIUtil;

import org.jetbrains.annotations.NotNull;

import java.awt.Color;
import java.awt.Component;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.List;

import javax.swing.Icon;
import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.UIManager;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.JTableHeader;
import javax.swing.table.TableColumn;
import javax.swing.table.TableColumnModel;

/**
 * UiUtILS
 *
 * @author CABBAGE
 * @since 2023/06/03
 */
public class UiUtils {
    /**
     * left indent
     */
    public static final String LEFT_INDENT = "  ";

    private static final Color TABLE_ODD_ROW_BACKGROUND = new JBColor(0xF9F9F9, 0X202022);

    private static final Color TABLE_EVEN_ROW_BACKGROUND = new JBColor(0xFFFFFF, 0X1A1B1C);

    private static final Color DEFAULT_PANEL = new JBColor(0xFFFFFF, 0X262626);

    private static final int TABLE_PREFERRED_ROW_HEIGHT = 30;

    /**
     * UiUtils
     */
    public UiUtils() {
    }

    /**
     * set table different color in two lines
     *
     * @param table the table need to be set
     */
    public static void setTableColumnColor(JTable table) {
        if (table == null) {
            return;
        }
        DefaultTableCellRenderer tcr = new DefaultTableCellRenderer() {
            private static final long serialVersionUID = 1L;

            /**
             * get table cell render component
             *
             * @param table JTable
             * @param value object
             * @param isSelected true/false
             * @param hasFocus true/false
             * @param row row index
             * @param column column index
             * @return component
             */
            public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected,
                                                           boolean hasFocus, int row, int column) {
                if (table == null) {
                    return this;
                }
                super.getTableCellRendererComponent(table, value, isSelected, hasFocus, row, column);
                int maxPreferredHeight = TABLE_PREFERRED_ROW_HEIGHT;
                table.setRowMargin(0);

                for (int i = 0; i < table.getColumnCount(); i++) {
                    this.setText(table.getValueAt(row, i) == null ? "" : table.getValueAt(row, i).toString());
                    this.setSize(table.getColumnModel().getColumn(column).getWidth(), 0);
                    maxPreferredHeight = Math.max(maxPreferredHeight, getPreferredSize().height);
                }

                if (table.getRowHeight(row) < maxPreferredHeight) {
                    table.setRowHeight(row, maxPreferredHeight);
                }

                if (row % 2 == 0) {
                    setBackground(TABLE_EVEN_ROW_BACKGROUND);
                } else {
                    setBackground(TABLE_ODD_ROW_BACKGROUND);
                }

                if (isSelected) {
                    setBackground(UIUtil.getTableSelectionBackground(true));
                }

                this.setText(value == null ? "" : value.toString());
                this.setHorizontalAlignment(JLabel.LEFT);
                this.setValue(LEFT_INDENT + value);

                return this;
            }
        };
        for (int i = 0; i < table.getColumnCount(); i++) {
            table.getColumn(table.getColumnName(i)).setCellRenderer(tcr);
        }
    }

    /**
     * return icon according to current theme
     *
     * @param darkIcon    icon in dark theme
     * @param icon        icon in white theme
     * @return Icon       current color
     */
    public static Icon getJbIcon(Icon darkIcon, Icon icon) {
        return isDark() ? darkIcon : icon;
    }

    /**
     * return color according to current theme
     *
     * @param darkColor    color in dark theme
     * @param color        color in white theme
     * @return Color       current color
     */
    public static Color getJbColor(Color darkColor, Color color) {
        return isDark() ? darkColor : color;
    }

    /**
     * judge whether the theme is dark
     *
     * @return is dark or not
     */
    public static boolean isDark() {
        return UIUtil.isUnderDarcula() || UIManager.getLookAndFeel().getName().contains("Dark");
    }

    static class ColumnWidth implements Comparable<ColumnWidth> {
        int id;

        int width;

        public ColumnWidth(int id, int width) {
            this.id = id;
            this.width = width;
        }

        @Override
        public int compareTo(@NotNull ColumnWidth other) {
            if (this.width > other.width) {
                return 1;
            } else {
                return -1;
            }
        }
    }

    private static int getColumnMaxNeedWidth(JTable jTable, TableColumn tableColumn, int columnIndex) {
        int maxWidth = jTable.getTableHeader()
                .getDefaultRenderer()
                .getTableCellRendererComponent(jTable, tableColumn.getIdentifier(), false, false, -1, columnIndex)
                .getPreferredSize().width;
        int rowCount = jTable.getRowCount();
        JTableHeader header = jTable.getTableHeader();
        int col = header.getColumnModel().getColumnIndex(tableColumn.getIdentifier());
        for (int row = 0; row < rowCount; row++) {
            int preferredWidth = (int) jTable.getCellRenderer(row, col)
                    .getTableCellRendererComponent(jTable, jTable.getValueAt(row, col), false, false, row, col)
                    .getPreferredSize()
                    .getWidth();
            maxWidth = Math.max(maxWidth, preferredWidth);
        }
        return maxWidth;
    }

    /**
     *  fit column width
     *
     * @param tableJscrollPane table scroll pane
     * @param table table
     */
    public static void fitColumnWidth(JScrollPane tableJscrollPane, JTable table) {
        if (tableJscrollPane == null || table == null) {
            return;
        }
        int width = tableJscrollPane.getSize().width;
        int contentWidthSum = 0;
        Enumeration<TableColumn> columns = table.getColumnModel().getColumns();
        List<ColumnWidth> columnWidthList = new ArrayList<>();
        int col = 0;
        while (columns.hasMoreElements()) {
            TableColumn column = columns.nextElement();
            int contentMaxNeedWidth = getColumnMaxNeedWidth(table, column, col);
            contentWidthSum += contentMaxNeedWidth;
            columnWidthList.add(new ColumnWidth(col, contentMaxNeedWidth));
            col++;
        }
        columnWidthList.sort(Comparator.naturalOrder());

        JTableHeader header = table.getTableHeader();
        TableColumnModel tableColumnModel = table.getColumnModel();
        int remainCount = table.getColumnCount();
        if (remainCount == 0) {
            return;
        }
        int remainMargin = width - contentWidthSum;
        int maxWidth = (int) (((double)width / remainCount) * 1.5);
        for (ColumnWidth columnWidth : columnWidthList) {
            TableColumn column = tableColumnModel.getColumn(columnWidth.id);
            header.setResizingColumn(column);
            int margin = remainMargin;
            if (remainCount > 1) {
                margin = Math.min((int) (columnWidth.width * 0.2), remainMargin / remainCount);
            }
            column.setWidth(Math.min(margin + columnWidth.width, maxWidth));
            remainMargin -= margin;
            remainCount--;
        }
    }

    /**
     * set background color
     *
     * @param components components to be set background color
     */
    public static void setBackgroundColor(JComponent[] components) {
        for (Component component : components) {
            component.setBackground(DEFAULT_PANEL);
        }
    }
}
