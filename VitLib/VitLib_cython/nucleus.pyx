# nucleus.pyx
from typing import Optional
import warnings

import cv2
import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import prange
from libcpp.vector cimport vector as cpp_vector
from libcpp.set cimport set as cpp_set

cdef extern from "<float.h>":
    const float FLT_MAX

from .common import small_area_reduction, detect_deleted_area_candidates, extract_threshold_values

DTYPE = np.uint8
ctypedef cnp.uint8_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[cnp.float32_t, ndim=1] calc_contour_areas(cnp.ndarray[DTYPE_t, ndim=2] img):
    """二値化画像から輪郭の面積を計算する関数

    Args:
        img (np.ndarray): 二値化画像(画素値: 0, 255)

    Returns:
        np.ndarray: 各検出された輪郭の面積（float32型）の1次元配列

    Example:
        >>> import cv2, numpy as np
        >>> img = np.array([...], dtype=np.uint8)
        >>> areas = calc_contour_areas(img)
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
cpdef float calc_standard_nuclear_area(cnp.ndarray[DTYPE_t, ndim=2] ans_img, float lower_ratio=17, float higher_ratio=0):
    """標準的核面積を計算する関数

    Args:
        ans_img (np.ndarray): 二値化された画像。
        lower_ratio (float): 除外する面積の下位割合 (0-100)。
        higher_ratio (float): 除外する面積の上位割合 (0-100)。

    Returns:
        float: 標準的核面積

    Example:
        >>> import numpy as np
        >>> from VitLib.VitLib_python.nucleus import calc_standard_nuclear_area
        >>> ans_img = np.array([...], dtype=np.uint8)
        >>> calc_standard_nuclear_area(ans_img)

    Note:
        例としてlower_ratio=0.1, higher_ratio=0.1の場合、下位10%と上位10%の面積を除外した中間の80%の面積を使用して標準的核面積の計算を行う
        use_cython
    """
    cdef int ans_unique_len, out_lower_num, out_heigher_num, contours_len, i
    cdef cnp.ndarray[cnp.float32_t, ndim=1] area_size, sorted_area_size
    if lower_ratio + higher_ratio < 0 or lower_ratio + higher_ratio > 100:
        raise ValueError("lower_ratio + higher_ratio must be in the range of 0-100")

    ans_unique_len = len(np.unique(ans_img))
    if ans_unique_len != 2 and ans_unique_len != 1:
        warnings.warn("ans_imgは二値画像ではありません。閾値127で二値化を行います。", UserWarning)
        ans_img = cv2.threshold(ans_img, 127, 255, cv2.THRESH_BINARY)[1]

    area_size = calc_contour_areas(ans_img)
    contours_len = len(area_size)
    out_lower_num = int(contours_len*lower_ratio/100)
    out_heigher_num = int(contours_len*higher_ratio/100)
    sorted_area_size = np.sort(area_size)[out_lower_num:contours_len-out_heigher_num]
    return np.mean(sorted_area_size)
###

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dict make_nuclear_evaluate_images(cnp.ndarray[DTYPE_t, ndim=2] ans_img, cnp.ndarray[DTYPE_t, ndim=3] bf_img, float care_rate=75, float lower_ratio=17, float higher_ratio=0):
    """評価用画像を作成する関数

    Args:
        ans_img (np.ndarray): 二値化された画像。
        bf_img (np.ndarray): 明視野画像。
        care_rate (float): 除外する核の標準核面積に対する面積割合(%) (0-100)。
        lower_ratio (float): 除外する面積の下位割合(%) (0-100)。
        higher_ratio (float): 除外する面積の上位割合(%) (0-100)。

    Returns:
        dict: 評価用画像の辞書。

            - "eval_img": 評価用画像
            - "red_img": DontCare領域画像
            - "green_img": 正解領域画像

    Example:
        >>> import numpy as np
        >>> from VitLib.VitLib_python.nucleus import make_nuclear_evaluate_images
        >>> ans_img = np.array([[0, 0, 0, 0],
        ...                     [0, 255, 255, 0],
        ...                     [0, 255, 255, 0],
        ...                     [0, 0, 0, 0]], dtype=np.uint8)
        >>> bf_img = np.zeros((4, 4, 3), dtype=np.uint8)
        >>> make_nuclear_evaluate_images(ans_img, bf_img)

    Note:
        using cython
    """
    cdef int ans_unique_len = len(np.unique(ans_img))
    if ans_unique_len != 2 and ans_unique_len != 1:
        warnings.warn("ans_imgは二値画像ではありません。閾値127で二値化を行います。", UserWarning)
        ans_img = cv2.threshold(ans_img, 127, 255, cv2.THRESH_BINARY)[1]
    cdef tuple contours = cv2.findContours(ans_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    cdef float standard_nuclear_area = calc_standard_nuclear_area(ans_img, lower_ratio, higher_ratio)
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

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.float64_t[:] euclidean_distance(cnp.float64_t[:] ext_centroid, cnp.float64_t[:, :] ans_centroids) nogil:
    """抽出された核と正解核の重心間の最小距離およびそのインデックスを算出する関数

    Args:
        ext_centroid (list[float]): 抽出された核の重心 (例: [x, y])
        ans_centroids (list[list[float]]): 正解核の重心リスト (例: [[x1, y1], [x2, y2], ...])

    Returns:
        tuple: (最小距離のインデックス (int), 最小距離 (float))

    Example:
        >>> ext_centroid = [100.0, 150.0]
        >>> ans_centroids = [[90.0, 145.0], [120.0, 170.0]]
        >>> index, distance = euclidean_distance(ext_centroid, ans_centroids)

    Note:
        using cython
    """
    cdef float min_distance, distance
    cdef int min_index, i
    cdef int len_ans_centroids
    cdef cnp.float64_t[:, :] ans_centroid
    with gil:
        len_ans_centroids = len(ans_centroids)
    min_distance = FLT_MAX
    min_index = -1
    for i in range(len_ans_centroids):
        #distance = np.linalg.norm(np.array(ext_centroid) - np.array(ans_centroid))
        distance = (ext_centroid[0] - ans_centroids[i, 0])**2 + (ext_centroid[1] - ans_centroids[i, 1])**2
        if distance < min_distance:
            min_distance = distance
            min_index = i
    with gil:
        return np.array([min_index, np.sqrt(min_distance)], dtype=np.float64)
###

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dict evaluate_nuclear_prediction(cnp.ndarray[DTYPE_t, ndim=2] pred_img, cnp.ndarray[DTYPE_t, ndim=2] ans_img, float care_rate=75, float lower_ratio=17, float higher_ratio=0, int threshold=127, int del_area=0, str eval_mode="inclusion", int distance=5):
    """細胞核画像の評価を行う関数

    Args:
        pred_img (np.ndarray): 推論画像 (2値画像)
        ans_img (np.ndarray): 正解画像 (2値画像)
        care_rate (float): 除外する核の標準的核面積に対する面積割合(%) (0-100の範囲)  
        lower_ratio (float): 除外する面積の下位割合(%) (0-100の範囲)  
        higher_ratio (float): 除外する面積の上位割合(%) (0-100の範囲)  
        threshold (int): 二値化の閾値
        del_area (int): 小領域削除の面積閾値
        eval_mode (str): 評価モード ("inclusion", "proximity", "iou")
        distance (int): 評価モードが"proximity"の場合の距離(ピクセル)

    Returns:
        dict: 評価結果の辞書  
        
              - "precision": 適合率(float)
              - "recall": 再現率(float)
              - "fmeasure": F値(float)
              - "threshold": 使用した二値化の閾値(int)
              - "del_area": 使用した小領域削除の閾値(int)
              - "correct_num": 正解核数(int, デバッグ用)
              - "conformity_bottom": 適合率計算用の核数(int, デバッグ用)
              - "care_num": 注意対象核の数(int, デバッグ用)

    Example:
        >>> import numpy as np
        >>> from VitLib.VitLib_cython.nucleus import evaluate_nuclear_prediction
        >>> pred_img = np.array([[...]], dtype=np.uint8)
        >>> ans_img = np.array([[...]], dtype=np.uint8)
        >>> evaluate_nuclear_prediction(pred_img, ans_img)

    Note:
        using cython
    """
    cdef int ans_unique_len = len(np.unique(ans_img))
    cdef cnp.ndarray[DTYPE_t, ndim=2] care_img, no_care_img, pred_img_th, care_img_th, no_care_img_th, pred_img_th_del
    cdef dict eval_images
    cdef int pred_num, care_num, no_care_num, ext_no_care_num, correct_num
    cdef int i, care, no_care, conformity_bottom
    cdef cnp.ndarray[DTYPE_t, ndim=3] dummy_bf_img
    cdef cnp.int32_t[:, :] pred_labels, care_labels, no_care_labels, pred_stats, care_stats, no_care_stats
    cdef cnp.float64_t[:, :] pred_centroids, care_centroids, no_care_centroids
    cdef float precision, recall, fmeasure
    cdef list correct_list
    cdef cnp.float64_t[:] min_care_data, min_no_care_data
    cdef float min_care_distance, min_no_care_distance

    #cdef cnp.int64_t[:, :] care_num_cell
    #cdef cnp.ndarray[cnp.int64_t, ndim=1] pred_cell_labels
    cdef cnp.int64_t[:] overlap_area
    cdef double sum_iou, ave_iou
    cdef cpp_vector[cpp_set[int]] correct_set
    cdef int x, y, j, ROW, COLUMN, or_area

    if ans_unique_len != 2 and ans_unique_len != 1:
        warnings.warn("ans_imgは二値画像ではありません。閾値127で二値化を行います。", UserWarning)
        ans_img = cv2.threshold(ans_img, 127, 255, cv2.THRESH_BINARY)[1]
    
    # 正解画像の準備
    dummy_bf_img = np.zeros((ans_img.shape[0], ans_img.shape[1], 3), dtype=np.uint8)
    eval_images = make_nuclear_evaluate_images(ans_img, dummy_bf_img, care_rate, lower_ratio, higher_ratio)
    care_img = eval_images["green_img"]
    no_care_img = eval_images["red_img"]

    # 画像の二値化
    pred_img_th = cv2.threshold(pred_img, threshold, 255, cv2.THRESH_BINARY)[1]
    care_img_th = cv2.threshold(care_img, 127, 255, cv2.THRESH_BINARY)[1]
    no_care_img_th = cv2.threshold(no_care_img, 127, 255, cv2.THRESH_BINARY)[1]

    # 推論画像の小領域削除
    pred_img_th_del = small_area_reduction(pred_img_th, del_area)

    # ラベル処理
    pred_num, pred_labels, pred_stats, pred_centroids = cv2.connectedComponentsWithStats(pred_img_th_del)
    care_num, care_labels, care_stats, care_centroids = cv2.connectedComponentsWithStats(care_img_th)
    no_care_num, no_care_labels, no_care_stats, no_care_centroids = cv2.connectedComponentsWithStats(no_care_img_th)

    if eval_mode == "iou":
        sum_iou = 0
        correct_set = cpp_vector[cpp_set[int]](care_num)
        overlap_area = np.zeros(care_num, dtype=np.int64)
        ROW = pred_img_th_del.shape[0]
        COLUMN = pred_img_th_del.shape[1]
        for y in prange(ROW, nogil=True):
            for x in range(COLUMN):
                # 正解領域かつ抽出領域の場合
                if care_labels[y, x] != 0 and pred_labels[y, x] != 0:
                    correct_set[care_labels[y, x]].insert(pred_labels[y, x])
                    overlap_area[care_labels[y, x]] += 1
        for i in prange(1, care_num, nogil=True):
            if correct_set[i].size() != 0:
                or_area = care_stats[i, 4] - overlap_area[i]
                for j in correct_set[i]:
                    or_area += pred_stats[j, 4]
                sum_iou += float(overlap_area[i]) / float(or_area)
        ave_iou = sum_iou / (care_num-1) if care_num-1 != 0 else 0
        return {"iou": ave_iou, "threshold": threshold, "del_area": del_area}
    else:
        # 抽出判定
        correct_list = [] #抽出された正解核のラベル番号リスト(後に重複削除する)
        ext_no_care_num = 0 #抽出されたが考慮しない核の数
        if eval_mode == "inclusion":
            for i in range(1, pred_num):
                care = care_labels[int(pred_centroids[i, 1]+0.5), int(pred_centroids[i, 0]+0.5)] != 0
                no_care = no_care_labels[int(pred_centroids[i, 1]+0.5), int(pred_centroids[i, 0]+0.5)] != 0
                if care:
                    correct_list.append(care_labels[int(pred_centroids[i][1]+0.5), int(pred_centroids[i][0]+0.5)])
                elif no_care:
                    ext_no_care_num += 1
        elif eval_mode == "proximity":
            for i in range(1, pred_num):
                #min_care_index, min_care_distance = euclidean_distance(pred_centroids[i], care_centroids[1:])
                min_care_data = euclidean_distance(pred_centroids[i], care_centroids[1:])
                min_care_index = int(min_care_data[0])
                min_care_distance = min_care_data[1]
                #min_no_care_index, min_no_care_distance = euclidean_distance(pred_centroids[i], no_care_centroids[1:])
                min_no_care_data = euclidean_distance(pred_centroids[i], no_care_centroids[1:])
                min_no_care_index = int(min_no_care_data[0])
                min_no_care_distance = min_no_care_data[1]
                if min_care_distance < distance:
                    correct_list.append(min_care_index+1)
                elif min_no_care_distance < distance:
                    ext_no_care_num += 1
        else:
            raise ValueError("eval_mode must be 'inclusion', 'proximity' or 'iou'.")
        
        #重複削除
        correct_list = list(set(correct_list))

        correct_num = len(correct_list)

        #抽出された数(適合率計算用), -1は背景の分
        conformity_bottom = pred_num - 1 - ext_no_care_num

        #適合率
        precision = float(correct_num) / float(conformity_bottom) if conformity_bottom != 0 else 0

        #再現率
        recall = float(correct_num) / float(care_num-1) if care_num-1 != 0 else 0

        #F値
        fmeasure = (2*precision*recall) / (precision + recall) if precision + recall != 0 else 0    
        
        return {"precision": precision, "recall": recall, "fmeasure": fmeasure, "threshold": threshold, "del_area": del_area, 'correct_num': correct_num, 'conformity_bottom': conformity_bottom, 'care_num': care_num-1}
###

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.float64_t[:] thred_eval(DTYPE_t[:, :] pred_img_th_nwg_del, str eval_mode, int distance, int care_num, cnp.int32_t[:, :] care_labels, cnp.int32_t[:, :] care_stats, cnp.int32_t[:, :] no_care_labels, cnp.float64_t[:, :] care_centroids, cnp.float64_t[:, :] no_care_centroids) nogil:
    cdef int i, pred_num, ext_no_care_num, care, no_care, min_care_index, min_no_care_index
    cdef cnp.int32_t[:, :] pred_labels, pred_stats
    cdef cnp.float64_t[:, :] pred_centroids
    cdef cnp.float64_t[:] min_care_data, min_no_care_data, result
    cdef float min_care_distance, min_no_care_distance

    cdef float correct_num, conformity_bottom, precision, recall, fmeasure, return_care_num

    cdef cpp_set[int] correct_set

    #iou用
    cdef cnp.int64_t[:] overlap_area
    cdef double sum_iou, ave_iou
    cdef cpp_vector[cpp_set[int]] correct_set_iou
    cdef int x, y, j, ROW, COLUMN, or_area

    with gil:
        pred_num, pred_labels, pred_stats, pred_centroids = cv2.connectedComponentsWithStats(np.asarray(pred_img_th_nwg_del))
        result = np.zeros(6, dtype=np.float64)
        return_care_num = float(care_num-1)

    if eval_mode == "iou":
        sum_iou = 0
        correct_set_iou = cpp_vector[cpp_set[int]](care_num)
        with gil:
            ROW = pred_img_th_nwg_del.shape[0]
            COLUMN = pred_img_th_nwg_del.shape[1]
            overlap_area = np.zeros(care_num, dtype=np.int64)
        
        for y in range(ROW):
            for x in range(COLUMN):
                if care_labels[y, x] != 0 and pred_labels[y, x] != 0:
                    correct_set_iou[care_labels[y, x]].insert(pred_labels[y, x])
                    overlap_area[care_labels[y, x]] += 1
        for i in range(1, care_num):
            if correct_set_iou[i].size() != 0:
                or_area = care_stats[i, 4] - overlap_area[i]
                for j in correct_set_iou[i]:
                    or_area += pred_stats[j, 4]
                sum_iou += float(overlap_area[i]) / float(or_area)
        ave_iou = sum_iou / return_care_num if return_care_num != 0 else 0
        result[0] = ave_iou
    else:
        ext_no_care_num = 0
        if eval_mode == "inclusion":
            for i in range(1, pred_num):
                care = care_labels[int(pred_centroids[i, 1]+0.5), int(pred_centroids[i, 0]+0.5)] != 0
                no_care = no_care_labels[int(pred_centroids[i, 1]+0.5), int(pred_centroids[i, 0]+0.5)] != 0
                if care:
                    correct_set.insert(care_labels[int(pred_centroids[i][1]+0.5), int(pred_centroids[i][0]+0.5)])
                elif no_care:
                    ext_no_care_num += 1
        elif eval_mode == "proximity":
            for i in range(1, pred_num):
                min_care_data = euclidean_distance(pred_centroids[i], care_centroids[1:])
                min_care_index = int(min_care_data[0])
                min_care_distance = min_care_data[1]
                min_no_care_data = euclidean_distance(pred_centroids[i], no_care_centroids[1:])
                min_no_care_index = int(min_no_care_data[0])
                min_no_care_distance = min_no_care_data[1]
                if min_care_distance < distance:
                    correct_set.insert(min_care_index+1)
                elif min_no_care_distance < distance:
                    ext_no_care_num += 1
        else:
            raise ValueError("eval_mode must be 'inclusion' or 'proximity'")

        # 合計数
        correct_num = correct_set.size()

        # 抽出された数(適合率計算用), -1は背景の分
        conformity_bottom = pred_num - 1 - ext_no_care_num

        # 適合率
        precision = correct_num / conformity_bottom if conformity_bottom != 0 else 0

        #適合率
        precision = correct_num / conformity_bottom if conformity_bottom != 0 else 0

        #再現率
        recall = correct_num / (care_num-1) if care_num-1 != 0 else 0

        #F値
        fmeasure = (2*precision*recall) / (precision + recall) if precision + recall != 0 else 0

        result[0] = precision
        result[1] = recall
        result[2] = fmeasure
        result[3] = correct_num
        result[4] = conformity_bottom
        result[5] = return_care_num

    return result

###

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[cnp.float64_t, ndim=2] evaluate_nuclear_prediction_range(cnp.ndarray[DTYPE_t, ndim=2] pred_img, cnp.ndarray[DTYPE_t, ndim=2] ans_img, float care_rate=75, float lower_ratio=17, float higher_ratio=0, int min_th=0, int max_th=255, int step_th=1, int min_area=0, object max_area=None, int step_area=1, str eval_mode="inclusion", int distance=5, int otsu=False, int verbose=False):
    """複数の条件(二値化閾値、小領域削除面積)を変えて細胞核の評価を行う関数.

    Args:
        pred_img (np.ndarray): 推論画像 (2値画像)
        ans_img (np.ndarray): 正解画像 (2値画像)
        care_rate (float): 除外する核の標準的核面積に対する面積割合(%) (0-100の範囲)  
        lower_ratio (float): 除外する面積の下位割合(%) (0-100の範囲)  
        higher_ratio (float): 除外する面積の上位割合(%) (0-100の範囲)  
        th_min (int): 二値化の閾値の最小値  
        max_th (int): 二値化の閾値の最大値  
        step_th (int): 二値化の閾値のステップ  
        min_area (int): 除外する面積の最小値  
        max_area (int): 除外する面積の最大値  
        step_area (int): 除外する面積のステップ  
        eval_mode (str): 評価モード ("inclusion", "proximity", "iou")
        distance (int): 評価モードが"proximity"の場合の距離(ピクセル)
        otsu (bool): Otsuの二値化を行うかどうか
        verbose (bool): 進捗表示を行うかどうか

    Returns:
        np.ndarray: 評価指標の配列. 
        
            - 0: threshold(int)
            - 1: del_area(int)
            - 2: iou(float, eval_mode="iou"のみ), precision(float, eval_mode="inclusion" or "proximity")
            - 3: なし(eval_mode="iou"のみ), recall(float, eval_mode="inclusion" or "proximity")
            - 4: なし(eval_mode="iou"のみ), fmeasure(float, eval_mode="inclusion" or "proximity")
            - 5: なし(eval_mode="iou"のみ), correct_num(int, eval_mode="inclusion" or "proximity")
            - 6: なし(eval_mode="iou"のみ), conformity_bottom(int, eval_mode="inclusion" or "proximity")
            - 7: なし(eval_mode="iou"のみ), care_num(int, eval_mode="inclusion" or "proximity")

    Example:
        >>> import numpy as np
        >>> from VitLib.VitLib_cython.nucleus import evaluate_nuclear_prediction_range
        >>> pred_img = np.array([[...]], dtype=np.uint8)
        >>> ans_img = np.array([[...]], dtype=np.uint8)
        >>> evaluate_nuclear_prediction_range(pred_img, ans_img)

    Note:
        評価対象毎に二値化や小領域削除処理が適用されます。
        using cython
    """
    cdef int ROW = pred_img.shape[0]
    cdef int COLUMN = pred_img.shape[1]
    cdef cnp.float64_t[:, :] result

    cdef int ans_unique_len = len(np.unique(ans_img))
    cdef cnp.ndarray[DTYPE_t, ndim=3] dummy_bf_img
    cdef dict eval_images
    cdef DTYPE_t[:, :] care_img, no_care_img
    
    cdef DTYPE_t[:, :, :] pred_img_th
    cdef list pred_img_th_list = []
    
    cdef DTYPE_t[:] threshold_list
    cdef cnp.int32_t[:] del_area_list
    cdef int threshold_list_length, del_area_list_length, th_index
    cdef list temp_threshold_list

    cdef DTYPE_t[:, :, :] pred_img_th_del
    cdef cnp.ndarray[DTYPE_t, ndim=3] pred_img_th_del_

    cdef int pred_num, care_num, no_care_num, search_index, search_size
    cdef cnp.int32_t[:, :] pred_labels, care_labels, no_care_labels, pred_stats, care_stats, no_care_stats
    cdef cnp.float64_t[:, :] pred_centroids, care_centroids, no_care_centroids

    cdef cnp.ndarray[cnp.int32_t, ndim=1] pred_num_list
    cdef cnp.ndarray[cnp.int32_t, ndim=3] pred_labels_list, pred_stats_list
    cdef cnp.ndarray[cnp.float64_t, ndim=3] pred_centroids_list

    if ans_unique_len != 2 and ans_unique_len != 1:
        warnings.warn("ans_imgは二値画像ではありません。閾値127で二値化を行います。", UserWarning)
        ans_img = cv2.threshold(ans_img, 127, 255, cv2.THRESH_BINARY)[1]

    # 正解画像の準備
    dummy_bf_img = np.zeros((ans_img.shape[0], ans_img.shape[1], 3), dtype=np.uint8)
    eval_images = make_nuclear_evaluate_images(ans_img, dummy_bf_img, care_rate, lower_ratio, higher_ratio)
    care_img = eval_images["green_img"]
    no_care_img = eval_images["red_img"]

    care_num, care_labels, care_stats, care_centroids = cv2.connectedComponentsWithStats(np.asarray(care_img))
    no_care_num, no_care_labels, no_care_stats, no_care_centroids = cv2.connectedComponentsWithStats(np.asarray(no_care_img))

    # 二値化の閾値リスト
    if otsu:
        threshold_list = np.array([cv2.threshold(pred_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]], dtype=DTYPE)
        threshold_list_length = 1
    else:
        threshold_list = extract_threshold_values(pred_img, min_th=min_th, max_th=max_th)[::step_th]
        threshold_list_length = threshold_list.shape[0]
        if threshold_list_length == 0:
            temp_threshold_list = [min_th, max_th]
            temp_threshold_list = list(set(temp_threshold_list))
            threshold_list = np.array(temp_threshold_list, dtype=DTYPE)
            threshold_list_length = threshold_list.shape[0]
    pred_img_th = np.empty((threshold_list_length, ROW, COLUMN), dtype=DTYPE)

    # まとめて二値化処理
    for th_index in range(threshold_list_length):
        if verbose:print(f'\rThreshold: {threshold_list[th_index]}', end=' '*10)
        pred_img_th_list.append(cv2.threshold(pred_img, threshold_list[th_index], 255, cv2.THRESH_BINARY)[1])
    pred_img_th = np.stack(pred_img_th_list)

    # 除外する面積リスト(resultのサイズの決定)
    search_size = 0
    for th_index in range(threshold_list_length):
        if verbose:print(f'\rSearch size: {search_size}', end=' '*10)
        search_size += detect_deleted_area_candidates(np.asarray(pred_img_th[th_index]), min_area, max_area)[::step_area].shape[0]
    if verbose:print(f'\rSearch size: {search_size}')
    result = np.empty((search_size, 8), dtype=np.float64)
    pred_img_th_del_ = np.empty((search_size, ROW, COLUMN), dtype=np.uint8)

    # 小領域削除画像の作成
    search_index = 0
    for th_index in range(threshold_list_length):
        del_area_list = detect_deleted_area_candidates(np.asarray(pred_img_th[th_index]), min_area, max_area)[::step_area]
        del_area_list_length = del_area_list.shape[0]
        for del_area_index in range(del_area_list_length):
            if verbose:print(f'\rThreshold: {threshold_list[th_index]}, Del area: {del_area_list[del_area_index]}', end=' '*10)
            pred_img_th_del_[search_index] = small_area_reduction(np.asarray(pred_img_th[th_index]), del_area_list[del_area_index])
            result[search_index, 0] = threshold_list[th_index]
            result[search_index, 1] = del_area_list[del_area_index]
            search_index += 1
    pred_img_th_del = pred_img_th_del_

    for search_index in prange(search_size, nogil=True):
        result[search_index, 2:8] = thred_eval(pred_img_th_del[search_index], eval_mode, distance, care_num, care_labels, care_stats, no_care_labels, care_centroids, no_care_centroids)
    return np.asarray(result)
###
