import numpy as np


def cosine_similarity(pta_data: np.ndarray, acl_data: np.ndarray):
    pta_data = pta_data.reshape(-1)
    acl_data = acl_data.reshape(-1)
    acl_data_norm = np.linalg.norm(acl_data, axis=-1)
    pta_data_norm = np.linalg.norm(pta_data, axis=-1)
    result = pta_data.dot(acl_data) / (acl_data_norm * pta_data_norm)
    return result


def max_relative_error(pta_data: np.ndarray, acl_data: np.ndarray):
    np.seterr(divide='ignore', invalid='ignore')
    relative_err = np.divide((acl_data - pta_data), pta_data)
    max_relative_err = np.max(np.abs(relative_err))
    return max_relative_err


def mean_relative_error(pta_data: np.ndarray, acl_data: np.ndarray):
    np.seterr(divide='ignore', invalid='ignore')
    relative_err = np.divide((acl_data - pta_data), pta_data)
    mean_relative_err = np.average(np.abs(relative_err))
    return mean_relative_err


def relative_euclidean_distance(pta_data: np.ndarray, acl_data: np.ndarray):
    np.seterr(divide='ignore', invalid='ignore')
    distance = np.linalg.norm(pta_data - acl_data)
    relative_distance = distance / np.linalg.norm(pta_data)
    return relative_distance


cmp_alg_map = {
    # "cosine_similarity": cosine_similarity,
    "max_relative_error": max_relative_error,
    "mean_relative_error": mean_relative_error,
    "relative_euclidean_distance": relative_euclidean_distance
}
