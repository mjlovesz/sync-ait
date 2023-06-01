package com.ascend.ait.ide.service;

import com.ascend.ait.ide.commonlib.util.safeCmd.CmdStrBuffer;

public class CompareService {

    public static final String modelService = " -m ";
    public static final String offlineService = " -om ";
    public static final String inputService = " -i ";
    public static final String outputService = " -o ";
    public static final String cannService = " -c ";
    public static final String inputShapeService = " -s ";
    public static final String dymShapeService = " -dr ";
    public static final String deviceService = " --d ";
    public static final String outputNodesService = " --output-nodes ";
    public static final String outputSizeService = " --output-size ";
    public static final String advisorService = " --advisor ";
    public static final String dumpService = " --dump ";
    public static final String convertService = " --convert ";

    public void pathAdd(CmdStrBuffer strBuffer, String service, String param) {
        if (!param.isEmpty()) {
            strBuffer.append(service).appendFilePath(param);
        }
    }

    public void strAdd(CmdStrBuffer strBuffer, String service, String param) {
        if (!param.isEmpty()) {
            strBuffer.append(service).append(param);
        }
    }

    public void statueAdd(CmdStrBuffer strBuffer, String service, boolean isOn) {
        if (!isOn) {
            strBuffer.append(service).append("false");
        } else {
            strBuffer.append(service).append("true");
        }
    }
}
