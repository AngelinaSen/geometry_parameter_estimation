import numpy as np


def calculate_rel_error(ref_img: np.ndarray, img: np.ndarray) -> float:
    """
    Function to compute relative error between given image <img> and reference image <ref_img>
    ||img - ref_img||_2 / ||ref_img||_2
    :param ref_img: reference image
    :param img: given image
    :return: relative error
    """

    return np.linalg.norm(img - ref_img) / np.linalg.norm(ref_img)

