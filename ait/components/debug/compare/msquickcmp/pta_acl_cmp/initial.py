import os
import shutil
import site

from msquickcmp.pta_acl_cmp.constant import AIT_CMP_TASK_DIR, AIT_CMP_TASK, AIT_CMP_TASK_PID, \
    LD_PRELOAD, ACLTRANSFORMER_SAVE_TENSOR_MAX, ACLTRANSFORMER_SAVE_TENSOR, MAX_TOKEN_NUM, AIT_DUMP_CLEAN


def init_aclcmp_task(dump_clean):
    os.environ[AIT_CMP_TASK_PID] = str(os.getpid())
    os.environ[AIT_CMP_TASK] = "1"
    os.environ[AIT_CMP_TASK_DIR] = os.getcwd()
    os.environ[AIT_DUMP_CLEAN] = str(dump_clean)
    ld_preload = os.getenv("LD_PRELOAD")
    ld_preload = ld_preload or ""
    save_tensor_so_path = os.path.join(site.getsitepackages()[0], "msquickcmp", "libsavetensor.so")
    os.environ[LD_PRELOAD] = save_tensor_so_path + ":" + ld_preload

    # 加速库获取内部数据轮数，设大一些，否则token数目较多时，可能获取不到后面的数据
    os.environ[ACLTRANSFORMER_SAVE_TENSOR_MAX] = MAX_TOKEN_NUM
    # 打开加速库dump数据开关
    os.environ[ACLTRANSFORMER_SAVE_TENSOR] = str(1)

    acl_map_file_dir = os.path.join('/tmp', str(os.getpid()))
    if not os.path.exists(acl_map_file_dir):
        os.mkdir(acl_map_file_dir)

    csv_result_dir = os.path.join(os.getenv(AIT_CMP_TASK_DIR), os.getenv(AIT_CMP_TASK_PID))
    if not os.path.exists(csv_result_dir):
        os.mkdir(csv_result_dir)


def clear_aclcmp_task():
    acl_map_file_dir = os.path.join('/tmp', str(os.getpid()))
    if os.path.exists(acl_map_file_dir):
        shutil.rmtree(acl_map_file_dir)