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

package com.huawei.ascend.ait.ide.optimizie.ui.step;

import com.intellij.openapi.application.ApplicationManager;
import com.intellij.openapi.project.Project;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.SwingUtilities;
import javax.swing.table.DefaultTableModel;
import java.awt.BorderLayout;
import java.awt.Color;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

/**
 *ShowResult
 *
 * @author cabbage
 * @since 2023/06/07
 */
public class ShowResult extends JTable {
    private static final Logger LOGGER = LoggerFactory.getLogger(ShowResult.class);
    private DefaultTableModel model;
    private static int width=1100;
    private static int height=800;
    private JTable table;
    private String[] columnNames = null;
    private JPanel root;
    private JPanel excel;
    private final Project project;

    /**
     * ShowResult
     *
     * @param project project
     * @param filepath filepath
     */
    public ShowResult(Project project, String filepath) {
        this.project = project;
        root.setSize(width, height);
        root.setVisible(true);
        JFrame f = new JFrame();
        f.setTitle("Compare");
        f.getContentPane().setLayout(new BorderLayout());

        File file = new File(filepath);
        load(f, file.getAbsolutePath());
        f.setSize(width, height);
        f.setVisible(true);
    }

    private void InsertValue(List<String> data) {
        Future<?> future = ApplicationManager.getApplication().executeOnPooledThread(() -> {
            Vector<String> value = null;
            for (String datum : data) {
                String[] da = datum.split(",");
                value = new Vector<String>(Arrays.asList(da));
                addRow(value);
                try {
                    Thread.sleep(10);
                } catch (InterruptedException e) {
                    LOGGER.error(e.getMessage());
                }
            }
        });
        tryCatch(future);
    }

    private void addRow(final Vector value) {
        Future<?> future = ApplicationManager.getApplication().executeOnPooledThread(() -> {
            Runnable runnable = () -> model.addRow(value);
            SwingUtilities.invokeLater(runnable);
        });
        tryCatch(future);
    }

    private void tryCatch(Future<?> future) {
        try {
            future.get();
        } catch (ExecutionException e) {
            LOGGER.error(e.toString());
        } catch (InterruptedException e) {
            LOGGER.error(e.toString());
            Thread.currentThread().interrupt();
        }
    }

    private static Map<String, List<String>> readCSV(String path) {
        Map<String, List<String>> map = new HashMap<String, List<String>>();
        try {
            FileInputStream file = new FileInputStream(path);
            InputStreamReader streamReader = new InputStreamReader(file, StandardCharsets.UTF_8);
            BufferedReader reader = new BufferedReader(streamReader);
            getMap(reader, map);
        } catch (IOException ex) {
            LOGGER.error("Failed to read the CSV file.");
        }
        return map;
    }

    private static void getMap(BufferedReader reader, Map<String, List<String>> map) throws IOException {
        List<String> title = Collections.synchronizedList(new ArrayList<String>());
        List<String> data = Collections.synchronizedList(new ArrayList<String>());
        int i = 0;
        String line = "";
        while ((line = reader.readLine()) != null) {
            line = line.replace("\"", "");
            if (i == 0) {
                title.add(line);
            } else {
                data.add(line);
            }
            i++;
        }
        map.put("title", title);
        map.put("data", data);
    }

    private void load(JFrame frame, String path) {
        Map<String, List<String>> csv = readCSV(path);
        List<String> title = csv.get("title");
        List<String> data = csv.get("data");
        columnNames = title.get(0).split(",");
        loadTable(frame, data);
        frame.revalidate();
    }

    private void loadTable(JFrame frame, List<String> data){
        model = new DefaultTableModel(columnNames, 0);
        table.setModel(model);
        table.setSelectionBackground(Color.orange);
        table.setAutoResizeMode(JTable.AUTO_RESIZE_ALL_COLUMNS);
        for (int i = 0; i < columnNames.length; i++) {
            table.getColumnModel().getColumn(i).setPreferredWidth(200);
        }
        InsertValue(data);
        JScrollPane scroll = new JScrollPane(table);
        frame.getContentPane().add(scroll, BorderLayout.CENTER);
    }
}