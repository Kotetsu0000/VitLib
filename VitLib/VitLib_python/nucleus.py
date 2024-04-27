import warnings

import cv2
import numpy as np

from .common import smallAreaReduction

def calc_contour_areas(img:np.ndarray) -> np.ndarray:
    """
    画像の面積のリストを取得する関数
    
    Args:
        img (np.ndarray): 二値化画像

    Returns:
        np.ndarray: 面積のリスト
    """
    contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    contours_len = len(contours)
    area_size = np.zeros(contours_len, dtype=np.float32)
    for i in range(contours_len):
        contour = contours[i]
        area_size[i] = cv2.contourArea(contour)
    return area_size

def calc_standard_nuclear_area(ans_img:np.ndarray, lower_ratio:float=17, heigher_ratio:float=0):
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
    if lower_ratio + heigher_ratio < 0 or lower_ratio + heigher_ratio > 100:
        raise ValueError("lower_ratio + heigher_ratio must be in the range of 0-100")

    ans_unique_len = len(np.unique(ans_img))
    if ans_unique_len != 2 and ans_unique_len != 1:
        warnings.warn("ans_imgは二値画像ではありません。閾値127で二値化を行います。", UserWarning)
        ans_img = cv2.threshold(ans_img, 127, 255, cv2.THRESH_BINARY)[1]

    contours = cv2.findContours(ans_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    area_size = [cv2.contourArea(contour) for contour in contours]
    out_lower_num = int(len(area_size)*lower_ratio/100)
    out_heigher_num = int(len(area_size)*heigher_ratio/100)
    sorted_area_size = sorted(area_size)[out_lower_num:len(area_size)-out_heigher_num]
    return np.mean(sorted_area_size)

def make_eval_images(ans_img:np.ndarray, bf_img:np.ndarray, care_rate:float=75, lower_ratio:float=17, heigher_ratio:float=0):
    """
    評価用画像を作成する関数

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
    contours = cv2.findContours(ans_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    standard_nuclear_area = calc_standard_nuclear_area(ans_img, lower_ratio, heigher_ratio)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return_dict = {}

    red = [i for i in contours if cv2.contourArea(i) < care_rate/100 * standard_nuclear_area]
    red_len = len(red)
    green = [i for i in contours if not(cv2.contourArea(i) < care_rate/100 * standard_nuclear_area)]
    green_len = len(green)

    eval_img = ans_img.copy()
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
