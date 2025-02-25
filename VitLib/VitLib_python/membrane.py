import cv2
import numpy as np

from .common import small_area_reduction

def NWG(img:np.ndarray, symmetric:bool=False) -> np.ndarray:
    """NWG細線化を行う. 渡す画像は黒背景(0)に白(255)で描画されている2値画像である必要がある(cv2の2値化処理処理した画像).
    
    参考文献 : https://www.sciencedirect.com/science/article/pii/016786559500121V

    Args:
        img (np.ndarray): 黒背景(0)に白(255)で描かれた2値画像
        symmetric (bool): 対称的な細線化処理を行う場合はTrue

    Returns:
        np.ndarray: 細線化された画像 (画素値: 0 または 255)

    Example:
        >>> import numpy as np
        >>> from VitLib import NWG
        >>> img = np.array([[  0,   0,   0,   0,   0,   0,   0,   0],
        ...                 [  0,   0,   0,   0,   0,   0,   0,   0],
        ...                 [  0,   0,   0,   0,   0,   0,   0,   0],
        ...                 [  0,   0,   0, 255, 255, 255,   0,   0],
        ...                 [  0,   0,   0, 255, 255, 255,   0,   0],
        ...                 [  0,   0,   0, 255, 255, 255,   0,   0],
        ...                 [  0,   0,   0,   0,   0,   0,   0,   0],
        ...                 [  0,   0,   0,   0,   0,   0,   0,   0]])
        >>> NWG(img, symmetric=False)
        array([[  0,   0,   0,   0,   0,   0,   0,   0],
               [  0,   0,   0,   0,   0,   0,   0,   0],
               [  0,   0,   0,   0,   0,   0,   0,   0],
               [  0,   0,   0,   0,   0,   0,   0,   0],
               [  0,   0,   0,   0, 255,   0,   0,   0],
               [  0,   0,   0,   0,   0,   0,   0,   0],
               [  0,   0,   0,   0,   0,   0,   0,   0],
               [  0,   0,   0,   0,   0,   0,   0,   0]])

    Note:
        imgは2値画像を想定しており、[0, 255]の配列である必要がある。  
        cv2で二値化した画像を入れることで正常に動作する。
    """
    src = np.copy(img)//255

    # zero padding
    ROW, COLUMN = src.shape[0]+2, src.shape[1]+2
    pad = np.zeros((ROW, COLUMN))
    pad[1:ROW-1, 1:COLUMN-1] = src
    src = pad

    switch = True
    while True:
        r, c = src.nonzero()
        nei = np.array((src[r-1, c], src[r-1, c+1], src[r, c+1],
                        src[r+1, c+1], src[r+1, c], src[r+1, c-1],
                        src[r, c-1], src[r-1, c-1]))

        # condition 1
        nei_sum = np.sum(nei, axis=0)
        cond1 = np.logical_and(2 <= nei_sum, nei_sum <= 6)

        # condition 2
        nei = np.concatenate([nei, np.array([nei[0]])], axis=0)
        cond2 = np.zeros(nei.shape[1], dtype=np.uint8)
        for i in range(1, 9):
            cond2 += np.array(
                np.logical_and(nei[i-1] == 0, nei[i] == 1), dtype=np.uint8)
        cond2 = cond2 == 1
        nei = nei[0:8]

        # condition 3
        if symmetric:
            if switch:
                c3a = np.logical_and(
                    nei[0]+nei[1]+nei[2]+nei[5] == 0, nei[4]+nei[6] == 2)
                c3b = np.logical_and(
                    nei[2]+nei[3]+nei[4]+nei[7] == 0, nei[0]+nei[6] == 2)
                cond3 = np.logical_or(c3a, c3b)
            else:
                c3c = np.logical_and(
                    nei[1]+nei[4]+nei[5]+nei[6] == 0, nei[0]+nei[2] == 2)
                c3d = np.logical_and(
                    nei[0]+nei[3]+nei[6]+nei[7] == 0, nei[2]+nei[4] == 2)
                cond3 = np.logical_or(c3c, c3d)
        else:
            c3a = np.logical_and(
                nei[0]+nei[1]+nei[2]+nei[5] == 0, nei[4]+nei[6] == 2)
            c3b = np.logical_and(
                nei[2]+nei[3]+nei[4]+nei[7] == 0, nei[0]+nei[6] == 2)
            cond3 = np.logical_or(c3a, c3b)    

        # condition 4
        if switch:
            cond4 = (nei[2]+nei[4])*nei[0]*nei[6] == 0
        else:
            cond4 = (nei[0]+nei[6])*nei[2]*nei[4] == 0

        cond = np.logical_and(cond1, np.logical_or(cond2, cond3))
        cond = np.logical_and(cond, cond4)
        if True in cond:
            switch = not switch
        else:
            return (src[1:ROW-1, 1:COLUMN-1]*255).astype(np.uint8)

        src[r[cond], c[cond]] = 0

def modify_line_width(img: np.ndarray, radius: int = 1) -> np.ndarray:
    """細線化された画像の線の太さを変更する関数

    Args:
        img (np.ndarray): 2値画像 (画素値: 0, 1)
        radius (int): 線の太さを調整する半径

    Returns:
        np.ndarray: 線の太さが変更された画像 (画素値: 0, 1)

    Example:
        >>> import numpy as np
        >>> from VitLib import modify_line_width
        >>> img = np.array([[  0,   0,   0,   0,   0,   0,   0,   0],
        ...                 [  0,   0,   0,   0,   0,   0,   0,   0],
        ...                 [  0,   0,   0,   0,   0,   0,   0,   0],
        ...                 [  0,   0,   0,   1,   1,   1,   0,   0],
        ...                 [  0,   0,   0,   1,   1,   1,   0,   0],
        ...                 [  0,   0,   0,   1,   1,   1,   0,   0],
        ...                 [  0,   0,   0,   0,   0,   0,   0,   0],
        ...                 [  0,   0,   0,   0,   0,   0,   0,   0]])
        >>> modify_line_width(img, radius=1)
        array([[  0,   0,   0,   0,   0,   0,   0,   0],
               [  0,   0,   0,   0,   0,   0,   0,   0],
               [  0,   0,   1,   1,   1,   1,   1,   0],
               [  0,   0,   1,   1,   1,   1,   1,   0],
               [  0,   0,   1,   1,   1,   1,   1,   0],
               [  0,   0,   1,   1,   1,   1,   1,   0],
               [  0,   0,   1,   1,   1,   1,   1,   0],
               [  0,   0,   0,   0,   0,   0,   0,   0]])

    Note:
        入力画像は必ず2値画像（0と1の値）である必要があります。
    """
    src = np.copy(img)
    r, c = np.nonzero(src)
    for r, c in zip(r, c):
        src = cv2.circle(src, (c, r), radius, 1, thickness=-1)
    return src

def evaluate_membrane_prediction(pred_img:np.ndarray, ans_img:np.ndarray, threshold:int=127, del_area:int=100, symmetric:bool=False, radius:int=3, otsu:bool=False) -> dict:
    """細胞膜画像の評価を行う関数.

    Args:
        pred_img (np.ndarray): 推論画像(画素値 : [0, 255])
        ans_img (np.ndarray): 正解画像(画素値 : [0, 255])
        threshold (int): 二値化の閾値(otus=Trueの場合は無視される)
        del_area (int): 小領域削除の閾値  
        symmetric (bool): NWG細線化の対称性  
        radius (int): 評価指標の計算に使用する半径
        otsu (bool): 二値化の閾値を自動で設定するかどうか 

    Returns:
        dict: 評価指標、閾値、小領域削除の閾値、半径を含む辞書

            - precision : 適合率(float)
            - recall : 再現率(float)
            - fmeasure : F値(float)
            - threshold : 二値化の閾値(int)
            - del_area : 小領域削除の閾値(int)
            - radius : 評価指標の計算に使用する半径(int)
            - membrane_length : 正解の細胞膜の長さ(int, デバッグ用)
            - tip_length : 膨張した推定結果の内部に含まれる細線化した正解の長さ(int, デバッグ用)
            - miss_length : 正解の細胞膜の長さから膨張した推定結果の内部に含まれる細線化した正解の長さを引いたもの(int, デバッグ用)

    Example:
        >>> import numpy as np
        >>> from VitLib import evaluate_membrane_prediction
        >>> pred_img = np.array([[  0,   0,   0,   0,   0,   0,   0,   0],
        ...                      [  0,   0,   0,   0,   0,   0,   0,   0],
        ...                      [  0,   0,   0,   0,   0,   0,   0,   0],
        ...                      [  0,   0,   0, 255, 255, 255,   0,   0],
        ...                      [  0,   0,   0, 255, 255, 255,   0,   0],
        ...                      [  0,   0,   0, 255, 255, 255,   0,   0],
        ...                      [  0,   0,   0,   0,   0,   0,   0,   0],
        ...                      [  0,   0,   0,   0,   0,   0,   0,   0]])
        >>> ans_img = np.array([[  0,   0,   0,   0,   0,   0,   0,   0],
        ...                      [  0,   0,   0,   0,   0,   0,   0,   0],
        ...                      [  0,   0,   0,   0,   0,   0,   0,   0],
        ...                      [  0,   0,   0, 255, 255, 255,   0,   0],
        ...                      [  0,   0,   0, 255, 255, 255,   0,   0],
        ...                      [  0,   0,   0, 255, 255, 255,   0,   0],
        ...                      [  0,   0,   0,   0,   0,   0,   0,   0],
        ...                      [  0,   0,   0,   0,   0,   0,   0,   0]])
        >>> evaluate_membrane_prediction(pred_img, ans_img, threshold=127, del_area=0, radius=3)
    """
    # 画像の二値化
    if otsu:
        threshold, pred_img_th = cv2.threshold(pred_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        pred_img_th = cv2.threshold(pred_img, threshold, 255, cv2.THRESH_BINARY)[1]
    ans_img_th = cv2.threshold(ans_img, 127, 255, cv2.THRESH_BINARY)[1]

    # NWG細線化
    pred_img_th_nwg = NWG(pred_img_th, symmetric=symmetric)
    ans_img_th_nwg = NWG(ans_img_th, symmetric=symmetric)

    # 正解画像と推論画像を[0,1]に変換
    pred_img_th_nwg = (pred_img_th_nwg//255).astype(np.uint8)
    ans_img_th_nwg = (ans_img_th_nwg//255).astype(np.uint8)

    # 小領域削除
    pred_img_th_nwg_del = small_area_reduction(pred_img_th_nwg, del_area)

    # 評価指標の計算
    ## 正解の細胞膜の長さ
    membrane_length = np.sum(ans_img_th_nwg)

    ## 推定画像と正解画像の線幅を変更
    pred_img_th_fattened = modify_line_width(pred_img_th_nwg_del, radius)
    ans_img_th_fattened = modify_line_width(ans_img_th_nwg, radius)

    ## 膨張した推定結果の内部に含まれる細線化した正解の長さ(target in predicted)
    tip_length = np.sum(np.logical_and(pred_img_th_fattened == 1, ans_img_th_nwg == 1))

    ## 正解の内部に含まれていない細線化した抽出結果の長さ
    miss_length = np.sum(np.logical_and(pred_img_th_nwg_del == 1, ans_img_th_fattened != 1))

    precision = 0 if tip_length + miss_length==0 else tip_length / (tip_length + miss_length)
    recall = 0 if membrane_length==0 else tip_length / membrane_length
    fmeasure = 0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)

    return {'precision':precision, 'recall':recall, 'fmeasure':fmeasure, 'threshold':threshold, 'del_area':del_area, 'radius':radius, 'membrane_length':membrane_length, 'tip_length':tip_length, 'miss_length':miss_length}

def evaluate_membrane_prediction_nwg(pred_img_th_nwg:np.ndarray, ans_img_th_nwg:np.ndarray, threshold:int=127, del_area:int=100, radius:int=3) -> dict:
    """細胞膜画像の評価を行う関数.

    Args:
        pred_img_th_nwg (np.ndarray): 予測画像. 二値化, NWG細線化済み(画素値: 0, 255)
        ans_img_th_nwg (np.ndarray): 正解画像. 二値化, NWG細線化済み(画素値: 0, 255)
        threshold (int): 二値化の閾値(返却用)
        del_area (int): 小領域削除の閾値
        radius (int): 評価指標の計算に使用する半径

    Returns:
        dict: 評価指標、閾値、小領域削除の閾値、半径を含む辞書

            - precision : 適合率(float)
            - recall : 再現率(float)
            - fmeasure : F値(float)
            - threshold : 二値化の閾値(int)
            - del_area : 小領域削除の閾値(int)
            - radius : 評価指標の計算に使用する半径(int)
            - membrane_length : 正解の細胞膜の長さ(int, デバッグ用)
            - tip_length : 膨張した推定結果の内部に含まれる細線化した正解の長さ(int, デバッグ用)
            - miss_length : 正解の細胞膜の長さから膨張した推定結果の内部に含まれる細線化した正解の長さを引いたもの(int, デバッグ用)

    Example:
        >>> import numpy as np
        >>> from VitLib import evaluate_membrane_prediction_nwg
        >>> pred_img = np.array([[  0,   0,   0,   0,   0,   0,   0,   0],
        ...                      [  0,   0,   0,   0,   0,   0,   0,   0],
        ...                      [  0,   0,   0,   0,   0,   0,   0,   0],
        ...                      [  0,   0,   0, 255, 255, 255,   0,   0],
        ...                      [  0,   0,   0, 255, 255, 255,   0,   0],
        ...                      [  0,   0,   0, 255, 255, 255,   0,   0],
        ...                      [  0,   0,   0,   0,   0,   0,   0,   0],
        ...                      [  0,   0,   0,   0,   0,   0,   0,   0]])
        >>> ans_img = np.array([[  0,   0,   0,   0,   0,   0,   0,   0],
        ...                     [  0,   0,   0,   0,   0,   0,   0,   0],
        ...                     [  0,   0,   0,   0,   0,   0,   0,   0],
        ...                     [  0,   0,   0, 255, 255, 255,   0,   0],
        ...                     [  0,   0,   0, 255, 255, 255,   0,   0],
        ...                     [  0,   0,   0, 255, 255, 255,   0,   0],
        ...                     [  0,   0,   0,   0,   0,   0,   0,   0],
        ...                     [  0,   0,   0,   0,   0,   0,   0,   0]])
        >>> evaluate_membrane_prediction_nwg(pred_img, ans_img, threshold=127, del_area=100, radius=3)
    """
    # 正解画像と推論画像を[0,1]に変換
    pred_img_th_nwg = (pred_img_th_nwg//255).astype(np.uint8)
    ans_img_th_nwg = (ans_img_th_nwg//255).astype(np.uint8)

    # 小領域削除
    pred_img_th_nwg_del = small_area_reduction(pred_img_th_nwg, del_area)

    # 評価指標の計算
    ## 正解の細胞膜の長さ
    membrane_length = np.sum(ans_img_th_nwg)

    ## 推定画像と正解画像の線幅を変更
    pred_img_th_fattened = modify_line_width(pred_img_th_nwg_del, radius)
    ans_img_th_fattened = modify_line_width(ans_img_th_nwg, radius)

    ## 膨張した推定結果の内部に含まれる細線化した正解の長さ(target in predicted)
    tip_length = np.sum(np.logical_and(pred_img_th_fattened == 1, ans_img_th_nwg == 1))

    ## 正解の内部に含まれていない細線化した抽出結果の長さ
    miss_length = np.sum(np.logical_and(pred_img_th_nwg_del == 1, ans_img_th_fattened != 1))

    precision = 0 if tip_length + miss_length==0 else tip_length / (tip_length + miss_length)
    recall = 0 if membrane_length==0 else tip_length / membrane_length
    fmeasure = 0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)

    return {'precision':precision, 'recall':recall, 'fmeasure':fmeasure, 'threshold':threshold, 'del_area':del_area, 'radius':radius, 'membrane_length':membrane_length, 'tip_length':tip_length, 'miss_length':miss_length}

def evaluate_membrane_prediction_range(pred_img:np.ndarray, ans_img:np.ndarray, radius:int=3, min_th:int=0, max_th:int=255, step_th:int=1, min_area:int=0, max_area:int=None, step_area:int=1, symmetric:bool=False, otsu:bool=False, verbose:bool=False) -> np.ndarray:
    """複数の条件(二値化閾値、小領域削除面積)を変えて細胞膜の評価を行う関数.

    Args:
        pred_img (np.ndarray): 予測画像.  
        ans_img (np.ndarray): 正解画像.  
        radius (int): 評価指標の計算に使用する半径.  
        min_th (int): 二値化の最小閾値.  
        max_th (int): 二値化の最大閾値.  
        step_th (int): 二値化の閾値のステップ.  
        min_area (int): 小領域削除の最小面積.  
        max_area (int): 小領域削除の最大面積.  
        step_area (int): 小領域削除の面積のステップ.  
        symmetric (bool): NWG細線化の対称性.  
        otsu (bool): 二値化の閾値を自動で設定するかどうか.  
        verbose (bool): 進捗表示を行うかどうか.  

    Returns:
        np.ndarray: 評価指標の配列. 
        
            - 0: threshold(int)
            - 1: del_area(int)
            - 2: precision(float)
            - 3: recall(float)
            - 4: fmeasure(float)
            - 5: membrane_length(int, デバッグ用)
            - 6: tip_length(int, デバッグ用)
            - 7: miss_length(int, デバッグ用)

    Example:
        >>> import numpy as np
        >>> from VitLib import evaluate_membrane_prediction_range
        >>> pred_img = np.array([[...]], dtype=np.uint8)
        >>> ans_img = np.array([[...]], dtype=np.uint8)
        >>> evaluate_membrane_prediction_range(pred_img, ans_img, radius=3, min_th=0, max_th=255, step_th=1, min_area=0, max_area=None, step_area=1, symmetric=False, otsu=False, verbose=False)
    
    Note:
        This function is only available in Cython.
    """

    ## Cythonでしか実行できない関数なのでエラーを出す。
    raise NotImplementedError("This function can only be executed in Cython.")

def evaluate_membrane_prediction_map(pred_img:np.ndarray, ans_img:np.ndarray, threshold:int=127, del_area:int=100, symmetric:bool=False, radius:int=3, otsu:bool=False, eval_size:int=256, map_step:int=1) -> np.ndarray:
    """細胞膜の精度マップを作成する関数

    Args:
        pred_img (np.ndarray): 予測画像(画素値: 0 ~ 255)
        ans_img (np.ndarray): 正解画像(画素値: 0 ~ 255)
        threshold (int): 二値化の閾値(otsu=Trueの場合は無視される)
        del_area (int): 小領域削除の閾値
        symmetric (bool): NWG細線化の対称性
        radius (int): 評価指標の計算に使用する半径
        otsu (bool): 二値化の閾値を自動で設定するかどうか
        eval_size (int): 評価画像のサイズ
        map_step (int): 評価画像のステップ

    Returns:
        np.ndarray: F値のマップ

    Example:
        >>> import cv2
        >>> from VitLib.VitLib_cython import membrane
        >>> pred_img = cv2.imread('tests/pred_img.png', cv2.IMREAD_GRAYSCALE)
        >>> ans_img = cv2.imread('tests/ans_img.png', cv2.IMREAD_GRAYSCALE)
        >>> membrane.evaluate_membrane_prediction_map(pred_img, ans_img, threshold=127, del_area=100, symmetric=False, radius=3, otsu=False, eval_size=256, map_step=1)
    
    Note:
        This function is only available in Cython.
    """

    ## Cythonでしか実行できない関数なのでエラーを出す。
    raise NotImplementedError("This function can only be executed in Cython.")
