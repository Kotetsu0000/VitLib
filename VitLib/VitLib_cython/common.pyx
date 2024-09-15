# common.pyx
import cv2
import numpy as np
cimport numpy as cnp
cimport cython

DTYPE = np.uint8
ctypedef cnp.uint8_t DTYPE_t

def small_area_reduction_nofix_old(img, area_th=100):
    """2値画像の小領域削除を行う.

    Args:
        img (np.ndarray): 2値画像.
        area_th (int): 面積の閾値.

    Returns:
        np.ndarray: 小領域削除後の2値画像.

    Example:
        >>> import numpy as np
        >>> from nwg_cython import small_area_reduction_nofix_old
        >>> img = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
        ...                 [0, 0, 0, 0, 0, 0, 0, 0],
        ...                 [0, 0, 0, 0, 0, 0, 0, 0],
        ...                 [0, 0, 0, 1, 1, 1, 0, 0],
        ...                 [0, 0, 0, 1, 1, 1, 0, 0],
        ...                 [0, 0, 0, 1, 1, 1, 0, 0],
        ...                 [0, 0, 0, 0, 0, 0, 0, 0],
        ...                 [0, 0, 0, 0, 0, 0, 0, 0]])
        >>> small_area_reduction_nofix_old(img, area_th=100)
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
        labeled_src = np.copy(labeled_img)
        labeled_src[labeled_src != label] = 0
        labeled_src[labeled_src == label] = 1

        if np.sum(labeled_src) < area_th:
            labeled_img[labeled_img == label] = 0

    labeled_img[labeled_img > 0] = 1
    return np.array(np.expand_dims(labeled_img, -1), dtype=np.uint8)
###

def small_area_reduction_nofix(img, area_th=100):
    """2値画像の小領域削除を行う.

    Args:
        img (np.ndarray): 2値画像.
        area_th (int): 面積の閾値.

    Returns:
        np.ndarray: 小領域削除後の2値画像.

    Example:
        >>> import numpy as np
        >>> from nwg_cython import small_area_reduction_nofix
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
        if label_sum < area_th:
            labeled_img[labeled_img == label] = 0
    
    labeled_img[labeled_img > 0] = 1
    return labeled_img.astype(np.uint8)
###

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_t, ndim=2] small_area_reduction(cnp.ndarray[DTYPE_t, ndim=2] img, int area_th=100):
    """2値画像の小領域削除を行う.

    Args:
        img (np.ndarray): 2値画像.
        area_th (int): 面積の閾値.

    Returns:
        np.ndarray: 小領域削除後の2値画像.
        
    Example:
        >>> import numpy as np
        >>> from nwg_cython import small_area_reduction
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
        if contours[label] >= area_th:
            return_img[labeled_img == label] = 1

    return return_img
###

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[cnp.uint64_t, ndim=1] detect_deleted_area_candidates(cnp.ndarray[DTYPE_t, ndim=2] img):
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

    Note:
        using cython.
    """
    cdef tuple labeling_result = cv2.connectedComponentsWithStats(img)
    cdef int num_of_labels = labeling_result[0]
    cdef cnp.ndarray[cnp.uint64_t, ndim=2] labeled_img = labeling_result[1].astype(np.uint64)
    cdef cnp.ndarray[cnp.uint64_t, ndim=1] contours = np.ones(num_of_labels, dtype=np.uint64)
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
cpdef cnp.ndarray[DTYPE_t, ndim=1]  extract_threshold_values(cnp.ndarray[DTYPE_t, ndim=2] img):
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

    Note:
        using cython.
    """
    cdef cnp.ndarray[DTYPE_t, ndim=1] img_flatten = np.ravel(img)
    cdef cnp.ndarray[DTYPE_t, ndim=1] img_unique = np.unique(img_flatten[img_flatten!=0]) - 1
    return img_unique
###
