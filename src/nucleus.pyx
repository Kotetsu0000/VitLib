# nucleus.pyx
import warnings

import cv2
import numpy as np
cimport numpy as cnp
cimport cython

from .common import smallAreaReduction

DTYPE = np.uint8
ctypedef cnp.uint8_t DTYPE_t

cpdef float calc_standard_nuclear_area(cnp.ndarray[DTYPE_t, ndim=2] ans_img, float lower_ratio=0, float heigher_ratio=0):
    """
    標準的核面積を計算する

    Args:
        ans_img (np.ndarray): 二値化画像
        lower_ratio (float): 除外する面積の下位割合(%) (0-100の範囲)
        heigher_ratio (float): 除外する面積の上位割合(%) (0-100の範囲)

    Returns:
        float: 標準的核面積
    
    Note:
        例としてlower_ratio=0.1, heigher_ratio=0.1の場合、下位10%と上位10%の面積を除外した中間の80%の面積を使用して標準的核面積の計算を行う
    """
    cdef tuple contours
    cdef int ans_unique_len, out_lower_num, out_heigher_num, contours_len
    cdef cnp.ndarray[cnp.int32_t, ndim=3] contour
    cdef cnp.ndarray[cnp.float32_t, ndim=1] area_size, sorted_area_size
    if lower_ratio + heigher_ratio < 0 or lower_ratio + heigher_ratio > 100:
        raise ValueError("lower_ratio + heigher_ratio must be in the range of 0-100")

    ans_unique_len = len(np.unique(ans_img))
    if ans_unique_len != 2 and ans_unique_len != 1:
        warnings.warn("ans_imgは二値画像ではありません。閾値127で二値化を行います。", UserWarning)
        ans_img = cv2.threshold(ans_img, 127, 255, cv2.THRESH_BINARY)[1]

    contours = cv2.findContours(ans_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    contours_len = len(contours)
    area_size = np.zeros(contours_len, dtype=np.float32)
    for i in range(contours_len):
        contour = contours[i]
        area_size[i] = cv2.contourArea(contour)
    out_lower_num = int(contours_len*lower_ratio/100)
    out_heigher_num = int(contours_len*heigher_ratio/100)
    sorted_area_size = np.sort(area_size)[out_lower_num:contours_len-out_heigher_num]
    return np.mean(sorted_area_size)