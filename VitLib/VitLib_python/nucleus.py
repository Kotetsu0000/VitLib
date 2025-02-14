import warnings

import cv2
import numpy as np

from .common import small_area_reduction

def calc_contour_areas(img:np.ndarray) -> np.ndarray:
    """与えられた二値化画像から全ての輪郭を検出し、各輪郭の面積を算出して返す関数です。

    Args:
        img (np.ndarray): 0と255の値のみを持つ二値化画像。対象物は255、背景は0となっている必要があります。

    Returns:
        np.ndarray: 各検出された輪郭の面積（float32型）の1次元NumPy配列。

    Examples:
        >>> import cv2, numpy as np
        >>> img = np.zeros((200, 200), dtype=np.uint8)
        >>> cv2.circle(img, (100, 100), 50, 255, -1)
        >>> areas = calc_contour_areas(img)
        >>> print(areas)  # 検出された輪郭の面積が表示される
    """
    contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    contours_len = len(contours)
    area_size = np.zeros(contours_len, dtype=np.float32)
    for i in range(contours_len):
        contour = contours[i]
        area_size[i] = cv2.contourArea(contour)
    return area_size

def calc_standard_nuclear_area(ans_img:np.ndarray, lower_ratio:float=17, higher_ratio:float=0):
    """標準的核面積を計算する

    Args:
        ans_img (np.ndarray): 二値化画像  
        lower_ratio (float): 除外する面積の下位割合(%) (0-100の範囲)  
        higher_ratio (float): 除外する面積の上位割合(%) (0-100の範囲)  

    Returns:
        float: 標準的核面積

    Examples:
        >>> import numpy as np
        >>> from VitLib.VitLib_cython.nucleus import calc_standard_nuclear_area
        >>> # 例: 2値化画像を作成（核は255、背景は0）
        >>> ans_img = np.array([[0, 0, 0, 0],
        ...                     [0, 255, 255, 0],
        ...                     [0, 255, 255, 0],
        ...                     [0, 0, 0, 0]], dtype=np.uint8)
        >>> # 下位10%と上位10%を除外して標準的核面積を計算する
        >>> area = calc_standard_nuclear_area(ans_img, lower_ratio=10, higher_ratio=10)
        >>> print("Standard nuclear area:", area)
    
    Note:
        例としてlower_ratio=0.1, higher_ratio=0.1の場合、下位10%と上位10%の面積を除外した中間の80%の面積を使用して標準的核面積の計算を行う
    """
    if lower_ratio + higher_ratio < 0 or lower_ratio + higher_ratio > 100:
        raise ValueError("lower_ratio + higher_ratio must be in the range of 0-100")

    ans_unique_len = len(np.unique(ans_img))
    if ans_unique_len != 2 and ans_unique_len != 1:
        warnings.warn("ans_imgは二値画像ではありません。閾値127で二値化を行います。", UserWarning)
        ans_img = cv2.threshold(ans_img, 127, 255, cv2.THRESH_BINARY)[1]

    contours = cv2.findContours(ans_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    area_size = [cv2.contourArea(contour) for contour in contours]
    out_lower_num = int(len(area_size)*lower_ratio/100)
    out_heigher_num = int(len(area_size)*higher_ratio/100)
    sorted_area_size = sorted(area_size)[out_lower_num:len(area_size)-out_heigher_num]
    return np.mean(sorted_area_size)

def make_nuclear_evaluate_images(ans_img:np.ndarray, bf_img:np.ndarray, care_rate:float=75, lower_ratio:float=17, higher_ratio:float=0):
    """評価用画像を作成する関数

    Args:
        ans_img (np.ndarray): 二値化画像  
        bf_img (np.ndarray): 明視野画像  
        care_rate (float): 除外する核の標準的核面積に対する面積割合(%) (0-100の範囲)  
        lower_ratio (float): 除外する面積の下位割合(%) (0-100の範囲)  
        higher_ratio (float): 除外する面積の上位割合(%) (0-100の範囲)  

    Returns:
        dict: 評価用画像の辞書。以下のキーを含む

            - "eval_img": 評価用画像
            - "red_img": DontCare領域画像
            - "green_img": 正解領域画像

    Examples:
        >>> import numpy as np
        >>> from VitLib.VitLib_cython.nucleus import make_nuclear_evaluate_images
        >>> # 簡単な二値化画像と明視野画像の作成
        >>> ans_img = np.array([[0, 0, 0, 0],
        ...                     [0, 255, 255, 0],
        ...                     [0, 255, 255, 0],
        ...                     [0, 0, 0, 0]], dtype=np.uint8)
        >>> bf_img = np.zeros((4, 4, 3), dtype=np.uint8)
        >>> # care_rate=75, lower_ratio=17, higher_ratio=0 を使用して評価用画像を作成
        >>> result = make_nuclear_evaluate_images(ans_img, bf_img, care_rate=75, lower_ratio=17, higher_ratio=0)
        >>> print(result.keys())  # dict_keys(['eval_img', 'red_img', 'green_img'])
    """
    ans_unique_len = len(np.unique(ans_img))
    if ans_unique_len != 2 and ans_unique_len != 1:
        warnings.warn("ans_imgは二値画像ではありません。閾値127で二値化を行います。", UserWarning)
        ans_img = cv2.threshold(ans_img, 127, 255, cv2.THRESH_BINARY)[1]

    contours = cv2.findContours(ans_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    standard_nuclear_area = calc_standard_nuclear_area(ans_img, lower_ratio, higher_ratio)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return_dict = {}

    red = [i for i in contours if cv2.contourArea(i) < care_rate/100 * standard_nuclear_area]
    red_len = len(red)
    green = [i for i in contours if not(cv2.contourArea(i) < care_rate/100 * standard_nuclear_area)]
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

def euclidean_distance(ext_centroid, ans_centroids):
    """重心の距離の最小値とそのインデックスを返す関数

    Args:
        ext_centroid (tuple of float): 抽出された核の重心 (例: (x, y))
        ans_centroids (list of tuple of float): 正解核の重心リスト (例: [(x1, y1), (x2, y2), ...])

    Returns:
        tuple: 最小距離のインデックスとその距離.
            - 最小距離のインデックス (int)
            - 最小距離 (float)

    Examples:
        >>> ext_centroid = [100.0, 150.0]
        >>> ans_centroids = [[90.0, 145.0], [120.0, 170.0]]
        >>> index, distance = euclidean_distance(ext_centroid, ans_centroids)
        >>> print("最小距離のインデックス:", index)
        最小距離のインデックス: 0
        >>> print("最小距離:", distance)
        最小距離: 11.18
    """
    min_distance = 2**31 - 1
    min_index = -1
    for i in range(len(ans_centroids)):
        #ans_centroid = ans_centroids[i]
        #distance = np.linalg.norm(np.array(ext_centroid) - np.array(ans_centroid))
        distance = (ext_centroid[0] - ans_centroids[i, 0])**2 + (ext_centroid[1] - ans_centroids[i, 1])**2
        if distance < min_distance:
            min_distance = distance
            min_index = i
    return min_index, np.sqrt(min_distance)

def evaluate_nuclear_prediction(pred_img:np.ndarray, ans_img:np.ndarray, care_rate:float=75, lower_ratio:float=17, higher_ratio:float=0, threshold:int=127, del_area:int=0, eval_mode="inclusion", distance:int=5):
    """細胞核画像の評価を行う関数.

    Args:
        pred_img (np.ndarray): 予測画像
        ans_img (np.ndarray): 正解画像
        care_rate (float): 除外する核の標準的核面積に対する面積割合(%) (0-100の範囲)
        lower_ratio (float): 除外する面積の下位割合(%) (0-100の範囲)
        higher_ratio (float): 除外する面積の上位割合(%) (0-100の範囲)
        threshold (int): 二値化の閾値
        del_area (int): 除外する面積
        eval_mode (str): 評価方法

            - "inclusion": 抽出された領域の重心が正解領域内にあれば正解、それ以外は不正解とするモード
            - "proximity": 抽出された領域の重心と最も近い正解領域の重心との距離が指定値以内であれば正解、そうでなければ不正解とするモード
            - "iou": IoUの計算を行うモード(正解画像と重なっている抽出された細胞核についてのIoU平均値)

        distance (int): 評価モードが"proximity"の場合の距離(ピクセル)  

    Returns:
        dict: 評価結果の辞書
        
            - precision (float): 適合率
            - recall (float): 再現率
            - fmeasure (float): F値
            - threshold (int): 二値化の閾値
            - del_area (int): 除外する面積

    Examples:
        >>> import numpy as np
        >>> # 例として、全て背景の画像を生成
        >>> pred_img = np.zeros((100, 100), dtype=np.uint8)
        >>> ans_img = np.zeros((100, 100), dtype=np.uint8)
        >>> # evaluate_nuclear_prediction関数をinclusionモードで実行
        >>> result = evaluate_nuclear_prediction(pred_img, ans_img, care_rate=75, lower_ratio=17, higher_ratio=0, threshold=127, del_area=0, eval_mode="inclusion", distance=5)
        >>> print("Precision:", result["precision"])
        >>> print("Recall:", result["recall"])
        >>> print("F-measure:", result["fmeasure"])
    """
    ans_unique_len = len(np.unique(ans_img))
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
        for i in range(1, care_num):
            care_num_cell = np.where(care_labels == i, 1, 0)
            pred_label_list = np.unique(pred_labels * care_num_cell)[1:]
            pred_cell = np.zeros_like(pred_labels)
            for j in pred_label_list:
                pred_cell += np.where(pred_labels == j, 1, 0)
            iou = np.sum(np.logical_and(care_num_cell == 1, pred_cell == 1)) / np.sum(np.logical_or(care_num_cell == 1, pred_cell == 1))
            sum_iou += iou
        iou = sum_iou / (care_num-1) if care_num-1 != 0 else 0
        return {"iou": iou, "threshold": threshold, "del_area": del_area}
    else:
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
            raise ValueError("eval_mode must be 'inclusion', 'proximity' or 'iou'.")
        
        #重複削除
        correct_list = list(set(correct_list))

        correct_num = len(correct_list)

        #抽出された数(適合率計算用), -1は背景の分
        conformity_bottom = pred_num - 1 - ext_no_care_num

        #適合率
        precision = correct_num / conformity_bottom if conformity_bottom != 0 else 0

        #再現率
        recall = correct_num / (care_num-1) if care_num-1 != 0 else 0

        #F値
        fmeasure = (2*precision*recall) / (precision + recall) if precision + recall != 0 else 0    
        
        return {"precision": precision, "recall": recall, "fmeasure": fmeasure, "threshold": threshold, "del_area": del_area, 'correct_num': correct_num, 'conformity_bottom': conformity_bottom, 'care_num': care_num-1}

def evaluate_nuclear_prediction_range(pred_img:np.ndarray, ans_img:np.ndarray, care_rate:float=75, lower_ratio:float=17, higher_ratio:float=0, min_th:int=0, max_th:int=255, step_th:int=1, min_area:int=0, max_area:int=None, step_area:int=1, eval_mode:str="inclusion", distance:int=5, otsu:bool=False, verbose:bool=False) -> np.ndarray:
    """複数の条件(二値化閾値、小領域削除面積)を変えて細胞核の評価を行う関数.

    Args:
        pred_img (np.ndarray): 予測画像  
        ans_img (np.ndarray): 正解画像  
        care_rate (float): 除外する核の標準的核面積に対する面積割合(%) (0-100の範囲)  
        lower_ratio (float): 除外する面積の下位割合(%) (0-100の範囲)  
        higher_ratio (float): 除外する面積の上位割合(%) (0-100の範囲)  
        th_min (int): 二値化の閾値の最小値  
        max_th (int): 二値化の閾値の最大値  
        step_th (int): 二値化の閾値のステップ  
        min_area (int): 除外する面積の最小値  
        max_area (int): 除外する面積の最大値  
        step_area (int): 除外する面積のステップ  
        eval_mode (str): 評価方法  

            - "inclusion": 抽出された領域の重心が正解領域の中にあれば正解、それ以外は不正解とするモード
            - "proximity": 抽出された領域の重心と最も近い正解領域の重心との距離が指定値以内であれば正解、そうでなければ不正解とするモード

        distance (int): 評価モードが"proximity"の場合の距離(ピクセル)
        otsu (bool): Otsuの二値化を行うかどうか
        verbose (bool): 進捗表示を行うかどうか

    Returns:
        np.ndarray: 評価指標の配列. 
        
            - 0: threshold
            - 1: del_area
            - 2: precision
            - 3: recall
            - 4: fmeasure
            - 5: correct_num
            - 6: conformity_bottom
            - 7: care_num

    Examples:
        >>> import numpy as np
        >>> from VitLib.VitLib_cython import nucleus
        >>> # 予測画像と正解画像の例としてランダムな配列を作成
        >>> pred_img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        >>> ans_img = (np.random.rand(256, 256) > 0.5).astype(np.uint8) * 255
        >>> result = nucleus.evaluate_nuclear_prediction_range(pred_img, ans_img)
        >>> print(result)

    Note:
        This function is only available in Cython.
    """

    ## Cythonでしか実行できない関数なのでエラーを出す。
    raise NotImplementedError("This function can only be executed in Cython.")
