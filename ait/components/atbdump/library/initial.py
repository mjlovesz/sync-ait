import os
import shutil
import site

from constant import ATB_SAVE_TENSOR_TIME, ATB_SAVE_TENSOR_IDS, ATB_SAVE_TENSOR_RUNNER, ATB_SAVE_TENSOR, \
    ATB_SAVE_TENSOR_RANGE, ATB_SAVE_TILING, LD_PRELOAD

def init_dump_task(args):
    if args.save_desc:
        os.environ[ATB_SAVE_TENSOR] = "2"
    else:
        os.environ[ATB_SAVE_TENSOR] = "1"
    
    os.environ[ATB_SAVE_TENSOR_TIME] = args.time
    os.environ[ATB_SAVE_TENSOR_IDS] = args.ids
    os.environ[ATB_SAVE_TENSOR_RUNNER] = args.opname
    
    os.environ[ATB_SAVE_TENSOR_RANGE] = args.range
    os.environ[ATB_SAVE_TILING] = args.tiling
    ld_preload = os.getenv(LD_PRELOAD)
    ld_preload = ld_preload or ""
    save_tensor_so_path = os.path.join(site.getsitepackages()[0], "msquickcmp", "libatb_probe.so")
    os.environ[LD_PRELOAD] = save_tensor_so_path + ":" + ld_preload


def clear_aclcmp_task():
    pass