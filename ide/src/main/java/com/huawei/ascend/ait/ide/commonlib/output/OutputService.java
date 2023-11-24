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

package com.huawei.ascend.ait.ide.commonlib.output;

import com.intellij.execution.impl.ConsoleViewImpl;
import com.intellij.execution.ui.ConsoleView;
import com.intellij.execution.ui.ConsoleViewContentType;
import com.intellij.openapi.application.ApplicationManager;
import com.intellij.openapi.application.ModalityState;
import com.intellij.openapi.project.Project;

import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Optional;
import java.util.TimeZone;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

/**
 * OutputService
 *
 * @author cabbage
 * @since 2023/06/03
 */
public class OutputService {
    private static final Logger LOGGER = LoggerFactory.getLogger(OutputService.class);

    private static final String SPACE = "  ";

    private ConcurrentLinkedDeque<OutputPair> messagesDeque = new ConcurrentLinkedDeque<>();

    private Project project;

    /**
     * OutputService
     *
     * @param project project
     */
    public OutputService(@NotNull Project project) {
        synchronized (TimeZone.class) {
            System.setProperty("user.timezone", "MindStudio-TimeZone");
            TimeZone.setDefault(null);
        }
        this.project = project;
    }

    /**
     * get output service instance
     *
     * @param project current project
     * @return a output service instance
     */
    public static OutputService getInstance(@NotNull Project project) {
        return project.getService(OutputService.class);
    }

    /**
     * active output windows
     */
    public void active() {
        OutputFactory.activate(project);
    }

    /**
     * sync show output windows
     */
    public void show() {
        OutputFactory.show(project);
    }

    /**
     * clear details output windows contents
     */
    public void clearDetail() {
        if (OutputFactory.getDetailConsoleViewmap().containsKey(project)) {
            OutputFactory.getDetailConsoleViewmap().get(project).clear();
        }
    }

    /**
     * clear output windows content
     */
    public void clear() {
        if (OutputFactory.getNormalConsoleViewmap().containsKey(project)) {
            OutputFactory.getNormalConsoleViewmap().get(project).clear();
        }
    }

    /**
     * scrollToEnd
     */
    public void scrollToEnd() {
        ConsoleView consoleView = OutputFactory.getNormalConsoleViewmap().get(project);
        if (consoleView == null) {
            return;
        }
        if (consoleView instanceof ConsoleViewImpl) {
            ApplicationManager.getApplication().invokeAndWait(((ConsoleViewImpl) consoleView)::scrollToEnd);
        }
    }

    /**
     * print message to output windows
     *
     * @param message message to print
     */
    public void print(@NotNull String message) {
        print(message, ConsoleViewContentType.NORMAL_OUTPUT);
    }

    /**
     * print error message to output windows
     *
     * @param message error message to print
     */
    public void warn(@NotNull String message) {
        print(message, ConsoleViewContentType.LOG_WARNING_OUTPUT);
    }

    /**
     * print error message to output windows
     *
     * @param message error message to print
     */
    public void error(@NotNull String message) {
        print(message, ConsoleViewContentType.LOG_ERROR_OUTPUT);
    }

    /**
     * print message to output windows
     *
     * @param message     message to print
     * @param contentType output content type
     */
    public void print(@NotNull String message, @NotNull ConsoleViewContentType contentType) {
        ApplicationManager.getApplication().invokeLater(() -> {
            printFlush(new OutputPair(message, OutputType.NORMAL, contentType));
            scrollToEnd();
        }, ModalityState.any());
    }

    /**
     * print message to output toolwindows detail content window
     *
     * @param message message to print
     */
    public void printToDetail(@NotNull String message) {
        printToDetail(message, ConsoleViewContentType.NORMAL_OUTPUT);
    }

    /**
     * print message to output toolwindows detail content window
     *
     * @param message     message to print
     * @param contentType output content type
     */
    public void printToDetail(@NotNull String message, @NotNull ConsoleViewContentType contentType) {
        ApplicationManager.getApplication().invokeLater(() -> {
            printFlush(new OutputPair(message, OutputType.DETAIL, contentType));
        }, ModalityState.any());
    }

    /**
     * get normal ConsoleView
     *
     * @return normal ConsoleView
     */
    public ConsoleView getNormalConsoleView() {
        return OutputFactory.getNormalConsoleViewmap().get(project);
    }

    /**
     * get detail ConsoleView
     *
     * @return detail ConsoleView
     */
    public ConsoleView getDetailConsoleView() {
        return OutputFactory.getDetailConsoleViewmap().get(project);
    }

    /**
     * print and flush output
     *
     * @param outputPair output pair
     */
    private void printFlush(@NotNull OutputPair outputPair) {
        messagesDeque.add(outputPair);
        flush();
    }

    /**
     * flush output deque with sync block
     */
    private synchronized void flush() {
        Future<?> future = ApplicationManager.getApplication().executeOnPooledThread(() -> {
            while (!messagesDeque.isEmpty()) {
                OutputPair outputPair = messagesDeque.poll();
                if (outputPair == null) {
                    return;
                }
                print(outputPair);
            }
        });

        try {
            future.get();
        } catch (ExecutionException e) {
            LOGGER.error(e.toString());
        } catch (InterruptedException e) {
            LOGGER.error(e.toString());
            Thread.currentThread().interrupt();
        }
    }

    /**
     * raw output
     *
     * @param outputPair output pair
     */
    private void print(@NotNull OutputPair outputPair) {
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        simpleDateFormat.setTimeZone(TimeZone.getDefault());
        String currentTime = simpleDateFormat.format(Calendar.getInstance().getTime());
        switch (outputPair.getOutputType()) {
            case DETAIL: {
                Optional<ConsoleView> optional = Optional.ofNullable(OutputFactory.getDetailConsoleViewmap())
                        .map(outputfACTORY -> outputfACTORY.get(project));
                if (optional.isEmpty()) {
                    LOGGER.error("detailConsoleViewmap is null");
                    return;
                }
                optional.get().print(currentTime + SPACE + outputPair.getText() + System.lineSeparator(),
                        outputPair.getContentType());
                break;
            }

            case NORMAL: {
                Optional<ConsoleView> optional = Optional.ofNullable(OutputFactory.getNormalConsoleViewmap())
                        .map(outputfACTORY -> outputfACTORY.get(project));
                if (optional.isEmpty()) {
                    LOGGER.error("normalConsoleViewmap is null");
                    return;
                }
                optional.get().print(currentTime + SPACE + outputPair.getText() + System.lineSeparator(),
                        outputPair.getContentType());
                break;
            }
        }
    }
}
