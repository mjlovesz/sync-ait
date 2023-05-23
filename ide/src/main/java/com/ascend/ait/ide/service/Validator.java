package com.ascend.ait.ide.service;

import com.google.protobuf.ServiceException;
import com.huawei.mindstudio.ssh.service.SshService;

public interface Validator {
    void verify() throws ServiceException;
    void verify(SshService service) throws ServiceException;
}
