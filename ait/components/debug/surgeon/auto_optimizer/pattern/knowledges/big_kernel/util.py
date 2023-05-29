START_ADD = "start_add"

Q_MATMUL = "q_matmul"
Q_MATMUL_W = "q_matmul_w"
Q_ADD = "q_add"
Q_ADD_B = "q_add_b"
Q_RESHAPE = "q_reshape"
Q_RESHAPE_S = "q_reshape_s"
Q_TRANSPOSE = "q_transpose"
Q_TRANSPOSE_PERM = "q_transpose_perm"

K_MATMUL = "k_matmul"
K_MATMUL_W = "k_matmul_w"
K_ADD = "k_add"
K_ADD_B = "k_add_b"
K_RESHAPE = "k_reshape"
K_RESHAPE_S = "k_reshape_s"
K_TRANSPOSE1 = "k_transpose1"
K_TRANSPOSE2 = "k_transpose2"
K_TRANSPOSE_PERM = "q_transpose_perm"
K_TRANSPOSE_PERM2 = "q_transpose2_perm"

V_MATMUL = "v_matmul"
V_MATMUL_W = "v_matmul_w"
V_ADD = "v_add"
V_ADD_B = "v_add_b"
V_RESHAPE = "v_reshape"
V_RESHAPE_S = "v_reshape_s"
V_TRANSPOSE = "v_transpose"
V_TRANSPOSE_PERM = "v_transpose_perm"

QK_MATMUL = "qk_matmul"
MUL = "mul"
MUL_B = "mul_b"
QK_MASK_ADD = "qk_mask_add"
QK_MASK_ADD_B = "qk_mask_add_b"
SOFTMAX = "softmax"

SCORE_V_MATMUL = "score_v_matmul"
TRANSPOSE = "transpose"
TRANSPOSE_PERM = "transpose_perm"
RESHAPE = "reshape"
RESHAPE_S = "reshape_s"
MATMUL = "matmul"
MATMUL_W = "matmul_w"
ADD = "add"
ADD_B = "add_b"

END_ADD = "end_add"
END_ADD_B = "end_add_b"

CONVERT_3DIMS_TO_4DIMS = "convert_3dims_to_4dims"

def get_k_2nd_perm(q_perm):
    dims = len(q_perm)
    k_transpose_perm2 = list(range(len(q_perm)))
    for axis in range(len(q_perm)):
        if dims - axis == 2:
            k_transpose_perm2[axis] = axis + 1
        if dims - axis == 1:
            k_transpose_perm2[axis] = axis - 1
    return k_transpose_perm2
