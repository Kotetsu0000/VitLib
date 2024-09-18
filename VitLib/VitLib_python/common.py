"""細胞膜と細胞核の共通関数をまとめたモジュール."""
import cv2
import numpy as np

def small_area_reduction(img:np.ndarray, area_th:int=100) -> np.ndarray:
    """2値画像の小領域削除を行う.

    Args:
        img (np.ndarray): 2値画像.
        area_th (int): 面積の閾値.

    Returns:
        np.ndarray: 小領域削除後の2値画像.

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

def detect_deleted_area_candidates(img:np.ndarray) -> np.ndarray:
    """2値画像の小領域の削除面積のリストを作成する関数.

    Args:
        img (np.ndarray): 2値画像.

    Returns:
        np.ndarray: 小領域の面積のリスト.

    Example:
        >>> import numpy as np
        >>> from nwg_cython import detect_deleted_area_candidates
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
    num_of_labels, labeled_img, _, _ = cv2.connectedComponentsWithStats(img)
    contours = np.ones(num_of_labels, dtype=np.uint64)
    ROW, COLUMN = img.shape
    for row in range(ROW):
        for column in range(COLUMN):
            label = labeled_img[row, column]
            contours[label] += 1
    contours[0] = 0
    contours = np.unique(contours)
    contours.sort()
    return contours

def extract_threshold_values(img:np.ndarray) -> np.ndarray:
    """画像から閾値を抽出する.
    
    Args:
        img (np.ndarray): 2値画像.

    Returns:
        np.ndarray: 画像から抽出した閾値のリスト.

    Examples:
        >>> a = np.array([  0,   0,   0,   0,   0,   0,   0,   0,   0],
        ...              [127, 127, 127, 127, 127, 127, 127, 127, 127],
        ...              [255, 255, 255, 255, 255, 255, 255, 255, 255], dtype=np.uint8)
        >>> extract_threshold_values(a)
        array([126, 254], dtype=uint8)
    """
    img_flatten = np.ravel(img)
    img_unique = np.unique(img_flatten[img_flatten!=0]) - 1
    return img_unique
