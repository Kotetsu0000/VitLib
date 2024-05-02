# nucleus.pyx
from typing import Optional
import warnings

import cv2
import numpy as np
cimport numpy as cnp
cimport cython

cdef extern from "<float.h>":
    const float FLT_MAX

from .common import smallAreaReduction

DTYPE = np.uint8
ctypedef cnp.uint8_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[cnp.float32_t, ndim=1] calc_contour_areas(cnp.ndarray[DTYPE_t, ndim=2] img):
    """画像の面積のリストを取得する関数
    
    Args:
        img (np.ndarray): 二値化画像

    Returns:
        np.ndarray: 面積のリスト
    """
    cdef int contours_len, i
    cdef tuple contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    cdef cnp.ndarray[cnp.float32_t, ndim=1] area_size
    cdef cnp.ndarray[cnp.int32_t, ndim=3] contour
    contours_len = len(contours)
    area_size = np.zeros(contours_len, dtype=np.float32)
    for i in range(contours_len):
        contour = contours[i]
        area_size[i] = cv2.contourArea(contour)
    return area_size
###

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float calc_standard_nuclear_area(cnp.ndarray[DTYPE_t, ndim=2] ans_img, float lower_ratio=17, float heigher_ratio=0):
    """標準的核面積を計算する

    Args:
        ans_img (np.ndarray): 二値化画像
        lower_ratio (float): 除外する面積の下位割合(%) (0-100の範囲)
        heigher_ratio (float): 除外する面積の上位割合(%) (0-100の範囲)

    Returns:
        float: 標準的核面積
    
    Note:
        例としてlower_ratio=0.1, heigher_ratio=0.1の場合、下位10%と上位10%の面積を除外した中間の80%の面積を使用して標準的核面積の計算を行う
        use_cython
    """
    cdef int ans_unique_len, out_lower_num, out_heigher_num, contours_len, i
    cdef cnp.ndarray[cnp.float32_t, ndim=1] area_size, sorted_area_size
    if lower_ratio + heigher_ratio < 0 or lower_ratio + heigher_ratio > 100:
        raise ValueError("lower_ratio + heigher_ratio must be in the range of 0-100")

    ans_unique_len = len(np.unique(ans_img))
    if ans_unique_len != 2 and ans_unique_len != 1:
        warnings.warn("ans_imgは二値画像ではありません。閾値127で二値化を行います。", UserWarning)
        ans_img = cv2.threshold(ans_img, 127, 255, cv2.THRESH_BINARY)[1]

    area_size = calc_contour_areas(ans_img)
    contours_len = len(area_size)
    out_lower_num = int(contours_len*lower_ratio/100)
    out_heigher_num = int(contours_len*heigher_ratio/100)
    sorted_area_size = np.sort(area_size)[out_lower_num:contours_len-out_heigher_num]
    return np.mean(sorted_area_size)
###

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dict make_eval_images(cnp.ndarray[DTYPE_t, ndim=2] ans_img, cnp.ndarray[DTYPE_t, ndim=3] bf_img, float care_rate=75, float lower_ratio=17, float heigher_ratio=0):
    """評価用画像を作成する関数

    Args:
        ans_img (np.ndarray): 二値化画像
        bf_img (np.ndarray): 明視野画像
        care_rate (float): 除外する核の標準的核面積に対する面積割合(%) (0-100の範囲)
        lower_ratio (float): 除外する面積の下位割合(%) (0-100の範囲)
        heigher_ratio (float): 除外する面積の上位割合(%) (0-100の範囲)

    Returns:
        dict: 評価用画像の辞書
            - "eval_img": 評価用画像
            - "red_img": DontCare領域画像
            - "green_img": 正解領域画像
    """
    cdef int ans_unique_len = len(np.unique(ans_img))
    if ans_unique_len != 2 and ans_unique_len != 1:
        warnings.warn("ans_imgは二値画像ではありません。閾値127で二値化を行います。", UserWarning)
        ans_img = cv2.threshold(ans_img, 127, 255, cv2.THRESH_BINARY)[1]
    cdef tuple contours = cv2.findContours(ans_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    cdef float standard_nuclear_area = calc_standard_nuclear_area(ans_img, lower_ratio, heigher_ratio)
    cdef list red, green
    cdef int red_len, green_len, contours_len, i
    cdef cnp.ndarray[DTYPE_t, ndim=2] red_img, green_img
    cdef cnp.ndarray[DTYPE_t, ndim=3] eval_img
    cdef cnp.ndarray[DTYPE_t, ndim=2] kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cdef dict return_dict = {}
    contours_len = len(contours)

    red = [contours[i] for i in range(contours_len) if cv2.contourArea(contours[i]) < care_rate/100 * standard_nuclear_area]
    red_len = len(red)
    green = [contours[i] for i in range(contours_len) if cv2.contourArea(contours[i]) >= care_rate/100 * standard_nuclear_area]
    green_len = len(green)

    eval_img = bf_img.copy()
    red_img = np.zeros_like(ans_img)
    green_img = np.zeros_like(ans_img)

    for i in range(red_len):
        cv2.fillPoly(eval_img, [red[i][:,0,:]], (0,0,255), lineType=cv2.LINE_8, shift=0)
        cv2.fillPoly(red_img, [red[i][:,0,:]], (255,255,255), lineType=cv2.LINE_8, shift=0)

    for i in range(green_len):
        cv2.fillPoly(eval_img, [green[i][:,0,:]], (0,255,0), lineType=cv2.LINE_8, shift=0)
        cv2.fillPoly(green_img, [green[i][:,0,:]], (255,255,255), lineType=cv2.LINE_8, shift=0)

    #オープニング
    red_img = cv2.morphologyEx(red_img, cv2.MORPH_OPEN, kernel, iterations=2)

    return_dict["eval_img"] = eval_img
    return_dict["red_img"] = red_img
    return_dict["green_img"] = green_img
    return return_dict
###

cpdef tuple euclidean_distance(cnp.ndarray[cnp.float64_t, ndim=1] ext_centroid, cnp.ndarray[cnp.float64_t, ndim=2] ans_centroids):
    """重心の距離の最小値とそのインデックスを返す関数

    Args:
        ext_centroid (tuple): 抽出された核の重心
        ans_centroids (list): 正解核の重心リスト

    Returns:
        tuple: 最小距離のインデックスとその距離
            - 最小距離のインデックス(int)
            - 最小距離(float)
    """
    cdef float min_distance, distance
    cdef int min_index, i
    cdef int len_ans_centroids = len(ans_centroids)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] ans_centroid
    min_distance = FLT_MAX
    min_index = -1
    for i in range(len_ans_centroids):
        ans_centroid = ans_centroids[i]
        distance = np.linalg.norm(np.array(ext_centroid) - np.array(ans_centroid))
        if distance < min_distance:
            min_distance = distance
            min_index = i
    return (min_index, min_distance)
###

cpdef dict evaluate_cell_prediction(cnp.ndarray[DTYPE_t, ndim=2] pred_img, cnp.ndarray[DTYPE_t, ndim=2] ans_img, float care_rate=75, float lower_ratio=17, float heigher_ratio=0, int threshold=127, int del_area=10, str eval_mode="inclusion", int distance=10):
    """細胞核画像の評価を行う関数.

    Args:
        pred_img (np.ndarray): 予測画像
        ans_img (np.ndarray): 正解画像
        care_rate (float): 除外する核の標準的核面積に対する面積割合(%) (0-100の範囲)
        lower_ratio (float): 除外する面積の下位割合(%) (0-100の範囲)
        heigher_ratio (float): 除外する面積の上位割合(%) (0-100の範囲)
        threshold (int): 二値化の閾値
        del_area (int): 除外する面積
        eval_mode (str): 評価方法
            - "inclusion": 抽出された領域の重心が正解領域の中にあれば正解、それ以外は不正解とするモード
            - "proximity": 抽出された領域の重心と最も近い正解領域の重心が指定した距離以内である場合を正解、そうでない場合を不正解とするモード
        distance (int): 評価モードが"proximity"の場合の距離(ピクセル)

    Returns:
        dict: 評価結果の辞書
            - precision (float): 適合率
            - recall (float): 再現率
            - fmeasure (float): F値
            - threshold (int): 二値化の閾値
            - del_area (int): 除外する面積
    """
    cdef int ans_unique_len = len(np.unique(ans_img))
    cdef cnp.ndarray[DTYPE_t, ndim=2] care_img, no_care_img, pred_img_th, care_img_th, no_care_img_th, pred_img_th_del
    cdef dict eval_images
    cdef int pred_num, care_num, no_care_num, ext_no_care_num
    cdef int i, care, no_care, conformity_bottom
    cdef cnp.ndarray[DTYPE_t, ndim=3] dummy_bf_img
    cdef cnp.ndarray[cnp.int32_t, ndim=2] pred_labels, care_labels, no_care_labels, pred_stats, care_stats, no_care_stats
    cdef cnp.ndarray[cnp.float64_t, ndim=2] pred_centroids, care_centroids, no_care_centroids
    cdef float precision, recall, fmeasure
    cdef list correct_list
    if ans_unique_len != 2 and ans_unique_len != 1:
        warnings.warn("ans_imgは二値画像ではありません。閾値127で二値化を行います。", UserWarning)
        ans_img = cv2.threshold(ans_img, 127, 255, cv2.THRESH_BINARY)[1]
    
    # 正解画像の準備
    dummy_bf_img = np.zeros((ans_img.shape[0], ans_img.shape[1], 3), dtype=np.uint8)
    eval_images = make_eval_images(ans_img, dummy_bf_img, care_rate, lower_ratio, heigher_ratio)
    care_img = eval_images["green_img"]
    no_care_img = eval_images["red_img"]

    # 画像の二値化
    pred_img_th = cv2.threshold(pred_img, threshold, 255, cv2.THRESH_BINARY)[1]
    care_img_th = cv2.threshold(care_img, 127, 255, cv2.THRESH_BINARY)[1]
    no_care_img_th = cv2.threshold(no_care_img, 127, 255, cv2.THRESH_BINARY)[1]

    # 推論画像の小領域削除
    pred_img_th_del = smallAreaReduction(pred_img_th, del_area)

    # ラベル処理
    pred_num, pred_labels, pred_stats, pred_centroids = cv2.connectedComponentsWithStats(pred_img_th_del)
    care_num, care_labels, care_stats, care_centroids = cv2.connectedComponentsWithStats(care_img_th)
    no_care_num, no_care_labels, no_care_stats, no_care_centroids = cv2.connectedComponentsWithStats(no_care_img_th)

    # 抽出判定
    correct_list = [] #抽出された正解核のラベル番号リスト(後に重複削除する)
    ext_no_care_num = 0 #抽出されたが考慮しない核の数
    if eval_mode == "inclusion":
        for i in range(1, pred_num):
            care = care_labels[int(pred_centroids[i][1]+0.5), int(pred_centroids[i][0]+0.5)] != 0
            no_care = no_care_labels[int(pred_centroids[i][1]+0.5), int(pred_centroids[i][0]+0.5)] != 0
            if care:
                correct_list.append(care_labels[int(pred_centroids[i][1]+0.5), int(pred_centroids[i][0]+0.5)])
            elif no_care:
                ext_no_care_num += 1
    elif eval_mode == "proximity":
        for i in range(1, pred_num):
            min_care_index, min_care_distance = euclidean_distance(pred_centroids[i], care_centroids[1:])
            min_no_care_index, min_no_care_distance = euclidean_distance(pred_centroids[i], no_care_centroids[1:])
            if min_care_distance < distance:
                correct_list.append(min_care_index+1)
            elif min_no_care_distance < distance:
                ext_no_care_num += 1
    else:
        raise ValueError("eval_mode must be 'inclusion' or 'proximity'")
    
    #重複削除
    correct_list = list(set(correct_list))

    #抽出された数(適合率計算用), -1は背景の分
    conformity_bottom = pred_num - 1 - ext_no_care_num

    #適合率
    precision = len(correct_list) / conformity_bottom if conformity_bottom != 0 else 0

    #再現率
    recall = len(correct_list) / (care_num-1) if care_num-1 != 0 else 0

    #F値
    fmeasure = (2*precision*recall) / (precision + recall) if precision + recall != 0 else 0    
    
    return {"precision": precision, "recall": recall, "fmeasure": fmeasure, "threshold": threshold, "del_area": del_area}
###
