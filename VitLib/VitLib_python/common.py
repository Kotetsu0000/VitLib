import cv2
import numpy as np

def small_area_reduction(img: np.ndarray, area_th: int = 100) -> np.ndarray:
    """2値画像の小領域削除を行う関数

    Args:
        img (np.ndarray): 2値画像
        area_th (int): 面積閾値。これ以下の面積を持つ領域が削除される

    Returns:
        np.ndarray: 小領域削除後の2値画像

    Example:
        >>> import numpy as np
        >>> from VitLib import small_area_reduction
        >>> img = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
        ...                 [0, 0, 0, 0, 0, 0, 0, 0],
        ...                 [0, 0, 0, 0, 0, 0, 0, 0],
        ...                 [0, 0, 0, 1, 1, 1, 0, 0],
        ...                 [0, 0, 0, 1, 1, 1, 0, 0],
        ...                 [0, 0, 0, 1, 1, 1, 0, 0],
        ...                 [0, 0, 0, 0, 0, 0, 0, 0],
        ...                 [0, 0, 0, 0, 0, 0, 0, 0]])
        >>> small_area_reduction(img, area_th=100)
        array([[0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0]])
    """
    src = np.copy(img)
    labeling_result = cv2.connectedComponentsWithStats(src)
    num_of_labels, labeled_img, contours, centroids = labeling_result

    for label in range(1, num_of_labels):
        label_sum = np.sum(labeled_img == label)
        if label_sum <= area_th:
            labeled_img[labeled_img == label] = 0
    
    labeled_img[labeled_img > 0] = 1
    return labeled_img.astype(np.uint8)

def detect_deleted_area_candidates(img: np.ndarray, min_area: int = 0, max_area: int = None) -> np.ndarray:
    """2値画像から小領域の削除候補となる面積リストを作成する関数

    Args:
        img (np.ndarray): 2値画像 (画素値: 0 またはその他)
        min_area (int): 最小面積の閾値
        max_area (int, optional): 最大面積の閾値 (Noneの場合は制限なし)

    Returns:
        np.ndarray: 小領域の面積リスト

    Example:
        >>> import numpy as np
        >>> from VitLib import detect_deleted_area_candidates
        >>> img = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
        ...                 [0, 0, 0, 0, 0, 0, 0, 0],
        ...                 [0, 0, 0, 0, 0, 0, 0, 0],
        ...                 [0, 0, 0, 1, 1, 1, 0, 0],
        ...                 [0, 0, 0, 1, 1, 1, 0, 0],
        ...                 [0, 0, 0, 1, 1, 1, 0, 0],
        ...                 [0, 0, 0, 0, 0, 0, 0, 0],
        ...                 [0, 0, 0, 0, 0, 1, 1, 1]])
        >>> detect_deleted_area_candidates(img)
        array([0, 3])
    """
    stats = cv2.connectedComponentsWithStats(img)[2][:, 4]
    stats[0] = 0
    if max_area is not None:
        stats = stats[stats<max_area]
    stats = stats[stats>=min_area]
    return np.unique(stats)

def extract_threshold_values(img: np.ndarray, min_th: int = 0, max_th: int = 255) -> np.ndarray:
    """グレースケール画像から閾値のリストを抽出する関数

    Args:
        img (np.ndarray): グレースケール画像
        min_th (int): 抽出する最小閾値 (デフォルト: 0)
        max_th (int): 抽出する最大閾値 (デフォルト: 255)

    Returns:
        np.ndarray: 抽出された閾値候補のリスト

    Example:
        >>> import numpy as np
        >>> from VitLib import extract_threshold_values
        >>> a = np.array([  0,   0,   0,   0,   0,   0,   0,   0,   0],
        ...              [127, 127, 127, 127, 127, 127, 127, 127, 127],
        ...              [255, 255, 255, 255, 255, 255, 255, 255, 255], dtype=np.uint8)
        >>> extract_threshold_values(a)
        array([126, 254], dtype=uint8)
    """
    th_list = np.unique(img[img!=0]) - 1
    return th_list[np.logical_and(th_list>=min_th, th_list<max_th)]
