# common.pyx
import cv2
import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import prange

DTYPE = np.uint8
ctypedef cnp.uint8_t DTYPE_t

def small_area_reduction_nofix(img, area_th=100):
    """2値画像の小領域削除を行う関数

    Args:
        img (np.ndarray): 2値画像
        area_th (int): 面積閾値。これ以下の面積を持つ領域が削除される

    Returns:
        np.ndarray: 小領域削除後の2値画像

    Example:
        >>> import numpy as np
        >>> from VitLib import small_area_reduction_nofix
        >>> img = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
        ...                 [0, 0, 0, 0, 0, 0, 0, 0],
        ...                 [0, 0, 0, 0, 0, 0, 0, 0],
        ...                 [0, 0, 0, 1, 1, 1, 0, 0],
        ...                 [0, 0, 0, 1, 1, 1, 0, 0],
        ...                 [0, 0, 0, 1, 1, 1, 0, 0],
        ...                 [0, 0, 0, 0, 0, 0, 0, 0],
        ...                 [0, 0, 0, 0, 0, 0, 0, 0]])
        >>> small_area_reduction_nofix(img, area_th=100)
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
###

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_t, ndim=2] small_area_reduction_old(cnp.ndarray[DTYPE_t, ndim=2] img, int area_th=100):
    """2値画像の小領域削除を行う関数

    Args:
        img (np.ndarray): 2値画像
        area_th (int): 面積閾値。これ以下の面積を持つ領域が削除される

    Returns:
        np.ndarray: 小領域削除後の2値画像
        
    Example:
        >>> import numpy as np
        >>> from VitLib import small_area_reduction_old
        >>> img = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
        ...                 [0, 0, 0, 0, 0, 0, 0, 0],
        ...                 [0, 0, 0, 0, 0, 0, 0, 0],
        ...                 [0, 0, 0, 1, 1, 1, 0, 0],
        ...                 [0, 0, 0, 1, 1, 1, 0, 0],
        ...                 [0, 0, 0, 1, 1, 1, 0, 0],
        ...                 [0, 0, 0, 0, 0, 0, 0, 0],
        ...                 [0, 0, 0, 0, 0, 0, 0, 0]])
        >>> small_area_reduction_old(img, area_th=100)
        array([[0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0]])
        
    Note:
        using cython.
    """
    cdef cnp.ndarray[DTYPE_t, ndim=2] src = np.copy(img)
    cdef tuple labeling_result = cv2.connectedComponentsWithStats(src)
    cdef int num_of_labels = labeling_result[0]
    cdef cnp.ndarray[DTYPE_t, ndim=2] return_img = np.zeros_like(src)
    cdef cnp.ndarray[cnp.uint64_t, ndim=2] labeled_img = labeling_result[1].astype(np.uint64)
    cdef cnp.ndarray[cnp.uint64_t, ndim=1] contours = np.zeros(num_of_labels, dtype=np.uint64)
    cdef int ROW = src.shape[0]
    cdef int COLUMN = src.shape[1]
    cdef int row, column, label
    
    for row in range(ROW):
        for column in range(COLUMN):
            label = labeled_img[row, column]
            contours[label] += 1

    for label in range(1, num_of_labels):
        if contours[label] > area_th:
            return_img[labeled_img == label] = 1

    return return_img
###

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_t, ndim=2] small_area_reduction(cnp.ndarray[DTYPE_t, ndim=2] img, int area_th=100):
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
        
    Note:
        using cython.
    """
    cdef int retval
    cdef cnp.int32_t[:, :] labeled_img, stats
    cdef cnp.float64_t[:, :] centroids
    cdef int ROW = img.shape[0]
    cdef int COLUMN = img.shape[1]
    cdef cnp.ndarray[DTYPE_t, ndim=2] return_img = np.zeros((ROW, COLUMN), dtype=np.uint8)
    cdef int row, column, label, i
    retval, labeled_img, stats, centroids = cv2.connectedComponentsWithStats(img)
    for row in prange(ROW, nogil=True):
        for column in range(COLUMN):
            label = labeled_img[row, column]
            if label == 0:
                continue
            elif stats[label, 4] > area_th:
                return_img[row, column] = 1
            else:
                return_img[row, column] = 0
    return return_img
###

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[cnp.uint64_t, ndim=1] detect_deleted_area_candidates_old(cnp.ndarray[DTYPE_t, ndim=2] img):
    """2値画像から小領域の削除候補となる面積リストを作成する関数

    Args:
        img (np.ndarray): 2値画像 (画素値: 0 またはその他)

    Returns:
        np.ndarray: 小領域の面積リスト

    Example:
        >>> import numpy as np
        >>> from VitLib import detect_deleted_area_candidates_old
        >>> img = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
        ...                 [0, 0, 0, 0, 0, 0, 0, 0],
        ...                 [0, 0, 0, 0, 0, 0, 0, 0],
        ...                 [0, 0, 0, 1, 1, 1, 0, 0],
        ...                 [0, 0, 0, 1, 1, 1, 0, 0],
        ...                 [0, 0, 0, 1, 1, 1, 0, 0],
        ...                 [0, 0, 0, 0, 0, 0, 0, 0],
        ...                 [0, 0, 0, 0, 0, 1, 1, 1]])
        >>> detect_deleted_area_candidates_old(img)
        array([0, 3])

    Note:
        using cython.
    """
    cdef tuple labeling_result = cv2.connectedComponentsWithStats(img)
    cdef int num_of_labels = labeling_result[0]
    cdef cnp.ndarray[cnp.uint64_t, ndim=2] labeled_img = labeling_result[1].astype(np.uint64)
    cdef cnp.ndarray[cnp.uint64_t, ndim=1] contours = np.zeros(num_of_labels, dtype=np.uint64)
    cdef int ROW = img.shape[0]
    cdef int COLUMN = img.shape[1]
    cdef int row, column, label, length
    for row in range(ROW):
        for column in range(COLUMN):
            label = labeled_img[row, column]
            contours[label] += 1
    contours[0] = 0
    contours = np.unique(contours)
    contours.sort()
    return contours
###

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[cnp.int32_t, ndim=1] detect_deleted_area_candidates(cnp.ndarray[DTYPE_t, ndim=2] img, int min_area=0, object max_area=None):
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

    Note:
        using cython.
    """
    cdef cnp.ndarray[cnp.int32_t, ndim=1] stats
    stats = cv2.connectedComponentsWithStats(img)[2][:, 4]
    stats[0] = 0
    if max_area is not None:
        stats = stats[stats<max_area]
    stats = stats[stats>=min_area]
    return np.unique(stats)
###

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_t, ndim=1] extract_threshold_values_old(cnp.ndarray[DTYPE_t, ndim=2] img):
    """グレースケール画像から閾値のリストを抽出する関数
    
    Args:
        img (np.ndarray): グレースケール画像

    Returns:
        np.ndarray: 画像から抽出した閾値のリスト

    Example:
        >>> import numpy as np
        >>> from VitLib import extract_threshold_values_old
        >>> a = np.array([  0,   0,   0,   0,   0,   0,   0,   0,   0],
        ...              [127, 127, 127, 127, 127, 127, 127, 127, 127],
        ...              [255, 255, 255, 255, 255, 255, 255, 255, 255], dtype=np.uint8)
        >>> extract_threshold_values_old(a)
        array([126, 254], dtype=uint8)

    Note:
        using cython.
    """
    cdef cnp.ndarray[DTYPE_t, ndim=1] img_flatten = np.ravel(img)
    cdef cnp.ndarray[DTYPE_t, ndim=1] img_unique = np.unique(img_flatten[img_flatten!=0]) - 1
    return img_unique
###

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_t, ndim=1] extract_threshold_values(cnp.ndarray[DTYPE_t, ndim=2] img, int min_th=0, int max_th=255):
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

    Note:
        using cython.
    """
    cdef cnp.ndarray[DTYPE_t, ndim=1] th_list = np.unique(img[img!=0]) - 1
    return th_list[np.logical_and(th_list>=min_th, th_list<max_th)]
###
