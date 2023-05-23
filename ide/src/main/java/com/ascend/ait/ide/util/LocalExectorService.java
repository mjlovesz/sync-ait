package com.ascend.ait.ide.util;

import com.ascend.ait.ide.service.Validator;
import com.google.protobuf.ServiceException;
import com.intellij.openapi.project.Project;

public class LocalExectorService {
    private final Project project;

    public LocalExectorService(Project project) {
        this.project = project;
    }

/*    public CmdStrBuffer executeVerifyAndMakeCommand(@NotNull Validator validator) throws ServiceException {
        validator.verify();
        return null;

    }*/

}
