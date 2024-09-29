# membrane.pyx
# distutils: language=c++
# distutils: extra_compile_args = ["-O3"]
# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=True

import cv2
import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import prange

from .common import small_area_reduction, detect_deleted_area_candidates, extract_threshold_values

DTYPE = np.uint8
ctypedef cnp.uint8_t DTYPE_t

def NWG_nofix(img, symmetric:bool=False):
    '''NWG細線化を行う. 渡す画像は黒背景(0)に白(255)で描画されている2値画像である必要がある(cv2の2値化処理処理した画像).
    
    参考文献 : https://www.sciencedirect.com/science/article/pii/016786559500121V

    Args:
        img (cnp.ndarray): 2次元の2値画像
        symmetric (bool): 対称的な細線化処理を行うかどうか

    Returns:
        cnp.ndarray: 細線化した画像

    Example:
        >>> import numpy as np
        >>> from nwg_cython import NWG_nofix
        >>> img = np.array([[  0,   0,   0,   0,   0,   0,   0,   0],
        ...                 [  0,   0,   0,   0,   0,   0,   0,   0],
        ...                 [  0,   0,   0,   0,   0,   0,   0,   0],
        ...                 [  0,   0,   0, 255, 255, 255,   0,   0],
        ...                 [  0,   0,   0, 255, 255, 255,   0,   0],
        ...                 [  0,   0,   0, 255, 255, 255,   0,   0],
        ...                 [  0,   0,   0,   0,   0,   0,   0,   0],
        ...                 [  0,   0,   0,   0,   0,   0,   0,   0]])
        >>> NWG_nofix(img, symmetric=False)
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
    '''
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
            return src[1:ROW-1, 1:COLUMN-1]*255

        src[r[cond], c[cond]] = 0
###

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_t, ndim=2] NWG_old_v0(cnp.ndarray[DTYPE_t, ndim=2] img, int symmetric=False):
    '''NWG細線化を行う. 渡す画像は黒背景(0)に白(255)で描画されている2値画像である必要がある(cv2の2値化処理処理した画像).
    
    参考文献 : https://www.sciencedirect.com/science/article/pii/016786559500121V

    Args:
        img (cnp.ndarray): 2次元の2値画像
        symmetric (bool): 対称的な細線化処理を行うかどうか

    Returns:
        cnp.ndarray: 細線化した画像

    Example:
        >>> import numpy as np
        >>> from nwg_cython import NWG_old
        >>> img = np.array([[  0,   0,   0,   0,   0,   0,   0,   0],
        ...                 [  0,   0,   0,   0,   0,   0,   0,   0],
        ...                 [  0,   0,   0,   0,   0,   0,   0,   0],
        ...                 [  0,   0,   0, 255, 255, 255,   0,   0],
        ...                 [  0,   0,   0, 255, 255, 255,   0,   0],
        ...                 [  0,   0,   0, 255, 255, 255,   0,   0],
        ...                 [  0,   0,   0,   0,   0,   0,   0,   0],
        ...                 [  0,   0,   0,   0,   0,   0,   0,   0]])
        >>> NWG_old(img)
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
    '''
    cdef cnp.ndarray[DTYPE_t, ndim=2] src = np.copy(img) // 255
    cdef int ROW = src.shape[0]+2
    cdef int COLUMN = src.shape[1]+2

    cdef cnp.ndarray[DTYPE_t, ndim=2] pad = np.zeros((ROW, COLUMN), dtype=DTYPE)

    cdef cnp.ndarray[DTYPE_t, ndim=2] nei
    cdef tuple temp
    cdef cnp.ndarray[cnp.int64_t, ndim=1] r, c
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] nei_sum
    cdef cnp.ndarray[DTYPE_t, ndim=1] cond1, cond2, c3a, c3b, c3c, c3d, cond3, cond4, cond
    cdef DTYPE_t switch = True

    pad[1:ROW-1, 1:COLUMN-1] = src
    src = pad
    
    while True:
        r, c = src.nonzero()
        nei = np.array((src[r-1, c], src[r-1, c+1], src[r, c+1],
                        src[r+1, c+1], src[r+1, c], src[r+1, c-1],
                        src[r, c-1], src[r-1, c-1]))

        # condition 1
        nei_sum = np.sum(nei, axis=0, dtype=DTYPE)
        cond1 = np.logical_and(2 <= nei_sum, nei_sum <= 6)

        # condition 2
        nei = np.concatenate([nei, np.array([nei[0]])], axis=0)
        cond2 = np.zeros(nei.shape[1], dtype=DTYPE)
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

        cond = np.logical_and(cond1, np.logical_or(cond2, cond3))#1<b<7 and (a==1 or c==1)
        cond = np.logical_and(cond, cond4)
        if True in cond:
            switch = not switch
        else:
            return src[1:ROW-1, 1:COLUMN-1]*255

        src[r[cond], c[cond]] = 0
###

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_t, ndim=2] NWG_old_v1(cnp.ndarray[DTYPE_t, ndim=2] img, int symmetric=False):
    '''NWG細線化を行う. 渡す画像は黒背景(0)に白(255)で描画されている2値画像である必要がある(cv2の2値化処理処理した画像).
    
    参考文献 : https://www.sciencedirect.com/science/article/pii/016786559500121V

    Args:
        img (cnp.ndarray): 2次元の2値画像
        symmetric (bool): 対称的な細線化処理を行うかどうか

    Returns:
        cnp.ndarray: 細線化した画像

    Example:
        >>> import numpy as np
        >>> from nwg_cython import NWG
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
            using cython.
    '''
    cdef cnp.ndarray[DTYPE_t, ndim=2] src = np.pad(np.copy(img) // 255, (1, 1), 'constant')
    cdef cnp.ndarray[DTYPE_t, ndim=2] Q
    cdef int ROW = src.shape[0]
    cdef int COLUMN = src.shape[1]

    cdef int x, y, g, h, i, c, d
    cdef int[9] p
    cdef int cond, cond1, cond2, cond3, cond4
    cdef int cond3a, cond3b, cond3c, cond3d

    # 初期化
    g = 1
    h = 1
    while h==1:
        Q = np.zeros_like(src)
        g = 1 - g
        h = 0

        for x in range(1,COLUMN-1):
            for y in range(1,ROW-1):
                if src[y,x]==1:
                    p[0] = src[y-1, x]
                    p[1] = src[y-1, x+1]
                    p[2] = src[y, x+1]
                    p[3] = src[y+1, x+1]
                    p[4] = src[y+1, x]
                    p[5] = src[y+1, x-1]
                    p[6] = src[y, x-1]
                    p[7] = src[y-1, x-1]
                    p[8] = p[0]

                    # condition 1
                    p_sum = p[0]+p[1]+p[2]+p[3]+p[4]+p[5]+p[6]+p[7]
                    cond1 = 1<p_sum and p_sum<7

                    # condition 2
                    a=0
                    for i in range(1, 9):
                        a += 1 if p[i-1]==0 and p[i]==1 else 0
                    cond2 = a==1

                    # condition 3
                    cond3a = p[0]+p[1]+p[2]+p[5]==0 and p[4]+p[6]==2
                    cond3b = p[2]+p[3]+p[4]+p[7]==0 and p[0]+p[6]==2
                    if symmetric:
                        cond3c = p[1]+p[4]+p[5]+p[6]==0 and p[0]+p[2]==2
                        cond3d = p[0]+p[3]+p[6]+p[7]==0 and p[2]+p[4]==2
                        c = cond3a or cond3b
                        d = cond3c or cond3d
                        cond3 = (1 - g)*c + g*d == 1
                    else:
                        cond3 = cond3a or cond3b
                        

                    # condition 4
                    if g==0:
                        cond4 = (p[2]+p[4])*p[0]*p[6] == 0
                    else:
                        cond4 = (p[0]+p[6])*p[2]*p[4] == 0
                    cond = cond1 and (cond2 or cond3) and cond4

                    if cond:
                        h=1
                    else:
                        Q[y,x]=1
        src = Q    
    return src[1:ROW-1, 1:COLUMN-1]*255
###

@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_t[:, :] NWG_(DTYPE_t[:, :] img, int symmetric=False, int multithread=True) noexcept nogil:
    cdef int ROW = img.shape[0]+2
    cdef int COLUMN = img.shape[1]+2
    cdef DTYPE_t[:, :] src, Q
    cdef int x, y

    with gil:
        src = np.ascontiguousarray(np.pad(np.copy(img) // 255, (1, 1), 'constant'))
        Q = np.zeros((ROW+2, COLUMN+2), dtype=np.uint8)

    # 初期化
    cdef int g = 1
    cdef int h = 1
    
    while h>=1:
        g = 1 - g
        h = 0

        if multithread:
            for x in prange(1,COLUMN-1, nogil=True):
                for y in range(1,ROW-1):
                    if src[y,x]==1:
                        if NWG_single(src, x, y, g, symmetric):
                            h+=1
                            Q[y,x]=0
                        else:
                            Q[y,x]=1
        else:
            for x in range(1,COLUMN-1):
                for y in range(1,ROW-1):
                    if src[y,x]==1:
                        if NWG_single(src, x, y, g, symmetric):
                            h+=1
                            Q[y,x]=0
                        else:
                            Q[y,x]=1
        with gil:
            src = np.copy(Q)
    return src[1:ROW-1, 1:COLUMN-1]
###

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int NWG_single(DTYPE_t[:, :] src, int x, int y, int g, int symmetric) noexcept nogil:
    cdef int p_sum, a, c, d, i
    cdef int cond, cond1, cond2, cond3, cond3a, cond3b, cond3c, cond3d, cond4
    cdef int[9] p
    p[0] = src[y-1, x]
    p[1] = src[y-1, x+1]
    p[2] = src[y, x+1]
    p[3] = src[y+1, x+1]
    p[4] = src[y+1, x]
    p[5] = src[y+1, x-1]
    p[6] = src[y, x-1]
    p[7] = src[y-1, x-1]
    p[8] = p[0]

    # condition 1
    p_sum = p[0]+p[1]+p[2]+p[3]+p[4]+p[5]+p[6]+p[7]
    cond1 = 1<p_sum and p_sum<7
    if not cond1:
        return 0

    # condition 2
    a=0
    for i in range(1, 9):
        a += 1 if p[i-1]==0 and p[i]==1 else 0
    cond2 = a==1

    # condition 3
    cond3a = p[0]+p[1]+p[2]+p[5]==0 and p[4]+p[6]==2
    cond3b = p[2]+p[3]+p[4]+p[7]==0 and p[0]+p[6]==2
    if symmetric:
        cond3c = p[1]+p[4]+p[5]+p[6]==0 and p[0]+p[2]==2
        cond3d = p[0]+p[3]+p[6]+p[7]==0 and p[2]+p[4]==2
        c = cond3a or cond3b
        d = cond3c or cond3d
        cond3 = (1 - g)*c + g*d == 1
    else:
        cond3 = cond3a or cond3b

    if not (cond2 or cond3):
        return 0

    # condition 4
    if g==0:
        cond4 = (p[2]+p[4])*p[0]*p[6] == 0
    else:
        cond4 = (p[0]+p[6])*p[2]*p[4] == 0
    
    if cond4:
        return 1
    else:
        return 0
###

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_t, ndim=2] NWG(DTYPE_t[:, :] img, int symmetric=False, int multithread=True):
    return np.asarray(NWG_(img, symmetric, multithread)) * 255
###

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_t, ndim=2] modify_line_width_old(cnp.ndarray[DTYPE_t, ndim=2] img, int radius=1):
    """細線化された画像の線の太さを変更する. 

    Args:
        img (np.ndarray): 2値画像.
        radius (int): 線の太さ.

    Returns:
        np.ndarray: 線の太さを変更した画像.

    Note:
        using cython.
    """
    cdef cnp.ndarray[DTYPE_t, ndim=2] src = np.copy(img)
    cdef r_list, c_list
    cdef int r, c
    r_list, c_list = np.nonzero(src)
    for r, c in zip(r_list, c_list):
        src = cv2.circle(src, (c, r), radius, 1, thickness=-1)
    return src
###

@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_t[:, :] modify_line_width_(DTYPE_t[:, :] img, int radius=1) nogil:
    """細線化された画像の線の太さを変更する. 

    Args:
        img (np.ndarray): 2値画像.
        radius (int): 線の太さ.

    Returns:
        np.ndarray: 線の太さを変更した画像.

    Note:
        using cython.
    """
    cdef int height = img.shape[0]
    cdef int width = img.shape[1]
    cdef DTYPE_t[:, :] output_image
    cdef int x, y, u, v
    with gil:
        output_image = np.zeros((height, width), dtype=DTYPE)

    for y in prange(height, nogil=True):
        for x in range(width):
            if img[y, x] == 1:
                output_image[y, x] = 1
                continue
            for v in range(max(0, y - radius), min(height, y + radius + 1)):
                for u in range(max(0, x - radius), min(width, x + radius + 1)):
                    if (u - x) * (u - x) + (v - y) * (v - y) <= ((radius) * (radius)) and img[v, u] == 1:
                        output_image[y, x] = 1
                        break
                if output_image[y, x] == 1:
                    break
    return output_image 
###

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_t, ndim=2] modify_line_width(DTYPE_t[:, :] img, int radius=1):
    """細線化された画像の線の太さを変更する. 

    Args:
        img (np.ndarray): 2値画像.
        radius (int): 線の太さ.

    Returns:
        np.ndarray: 線の太さを変更した画像.

    Note:
        using cython.
    """
    return np.asarray(modify_line_width_(img, radius))
###

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dict evaluate_membrane_prediction(cnp.ndarray[DTYPE_t, ndim=2] pred_img, cnp.ndarray[DTYPE_t, ndim=2] ans_img, int threshold=127, int del_area=100, int symmetric=False, int radius=3, int otsu=False):
    """細胞膜画像の評価を行う関数.

    Args:
        pred_img (np.ndarray): 予測画像.
        ans_img (np.ndarray): 正解画像.
        threshold (int): 二値化の閾値. otus=Trueの場合は無視される.
        del_area (int): 小領域削除の閾値.
        symmetric (bool): NWG細線化の対称性.
        radius (int): 評価指標の計算に使用する半径.
        otsu (bool): 二値化の閾値を自動で設定するかどうか.

    Returns:
        dict: 評価指標の辞書. precision, recall, fmeasure, threshold, del_areaをキーとする.

    Note:
        using cython.
    """
    cdef cnp.ndarray[DTYPE_t, ndim=2] pred_img_th, ans_img_th, pred_img_th_nwg, ans_img_th_nwg, pred_img_th_nwg_del, pred_img_th_fattened, ans_img_th_fattened
    cdef float membrane_length, tip_length, miss_length
    cdef float precision, recall, fmeasure
    cdef dict return_dict
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
    pred_img_th_nwg = (pred_img_th_nwg/255).astype(DTYPE)
    ans_img_th_nwg = (ans_img_th_nwg//255).astype(DTYPE)

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

    return_dict = {'precision':precision, 'recall':recall, 'fmeasure':fmeasure, 'threshold':threshold, 'del_area':del_area, 'radius':radius, 'membrane_length':int(membrane_length), 'tip_length':int(tip_length), 'miss_length':int(miss_length)}
    return return_dict
###

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dict evaluate_membrane_prediction_nwg(cnp.ndarray[DTYPE_t, ndim=2] pred_img_th_nwg, cnp.ndarray[DTYPE_t, ndim=2] ans_img_th_nwg, int threshold=127, int del_area=100, int radius=3):
    """細胞膜画像の評価を行う関数.

    Args:
        pred_img_th_nwg (np.ndarray): 予測画像. 二値化, NWG細線化済み.
        ans_img_th_nwg (np.ndarray): 正解画像.
        threshold (int): 二値化の閾値(返却用).
        del_area (int): 小領域削除の閾値.
        radius (int): 評価指標の計算に使用する半径.

    Returns:
        dict: 評価指標の辞書. precision, recall, fmeasure, threshold, del_areaをキーとする.

    Note:
        using cython.
    """
    cdef cnp.ndarray[DTYPE_t, ndim=2] pred_img_th_nwg_del, pred_img_th_fattened, ans_img_th_fattened
    cdef float membrane_length, tip_length, miss_length
    cdef float precision, recall, fmeasure
    cdef dict return_dict

    # 正解画像と推論画像を[0,1]に変換
    pred_img_th_nwg = (pred_img_th_nwg/255).astype(DTYPE)
    ans_img_th_nwg = (ans_img_th_nwg//255).astype(DTYPE)

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

    return_dict = {'precision':precision, 'recall':recall, 'fmeasure':fmeasure, 'threshold':threshold, 'del_area':del_area, 'radius':radius, 'membrane_length':int(membrane_length), 'tip_length':int(tip_length), 'miss_length':int(miss_length)}
    return return_dict
###

cdef cnp.float64_t[:] thred_eval(DTYPE_t[:, :] pred_img_th_nwg_del, DTYPE_t[:, :] ans_img_th_nwg, DTYPE_t[:, :] pred_img_th_fattened, DTYPE_t[:, :] ans_img_th_fattened) nogil:
    cdef float membrane_length, tip_length, miss_length, precision, recall, fmeasure
    cdef int ROW = pred_img_th_nwg_del.shape[0]
    cdef int COLUMN = pred_img_th_nwg_del.shape[1]
    cdef int y, x
    cdef cnp.float64_t[:] result
    membrane_length = 0
    tip_length = 0
    miss_length = 0
    for y in range(ROW):
        for x in range(COLUMN):
            membrane_length += ans_img_th_nwg[y, x]
            if pred_img_th_fattened[y, x] == 1 and ans_img_th_nwg[y, x] == 1:
                tip_length += 1
            if pred_img_th_nwg_del[y, x] == 1 and ans_img_th_fattened[y, x] != 1:
                miss_length += 1

    if tip_length + miss_length==0:
        precision = 0
    else:
        precision = tip_length / (tip_length + miss_length)

    if membrane_length==0:
        recall = 0
    else:
        recall = tip_length / membrane_length

    if precision + recall == 0:
        fmeasure = 0
    else:
        fmeasure = 2 * (precision * recall) / (precision + recall)
    
    with gil:
        #print(f'unique pred_img_th_nwg_del: {np.unique(np.asarray(pred_img_th_nwg_del))}, unique ans_img_th_fattened: {np.unique(np.asarray(ans_img_th_fattened))}')
        #print(f'membrane_length: {membrane_length}, tip_length: {tip_length}, miss_length: {miss_length}, {np.sum(np.logical_and(np.asarray(pred_img_th_nwg_del) == 1, np.asarray(ans_img_th_fattened) != 1))}')
        result = np.empty(6, dtype=np.float64)
        result[0] = precision
        result[1] = recall
        result[2] = fmeasure
        result[3] = membrane_length
        result[4] = tip_length
        result[5] = miss_length
    return result
###

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.float64_t[:, :] evaluate_membrane_prediction_range(cnp.ndarray[DTYPE_t, ndim=2] pred_img, cnp.ndarray[DTYPE_t, ndim=2] ans_img, int radius=3, int min_th=0, int max_th=255, int step_th=1, int min_area=0, object max_area=None, int step_area=1, int symmetric=False, int otsu=False, int verbose=False):
    cdef int ROW = pred_img.shape[0]
    cdef int COLUMN = pred_img.shape[1]
    
    cdef DTYPE_t[:] threshold_list
    cdef int threshold_list_length

    cdef DTYPE_t[:, :, :] pred_img_th
    cdef list pred_img_th_list = []

    cdef DTYPE_t[:, :, :] pred_img_th_nwg

    cdef cnp.ndarray[DTYPE_t, ndim=2] ans_img_th_nwg_ = np.asarray(NWG_(cv2.threshold(ans_img, 127, 255, cv2.THRESH_BINARY)[1]))
    cdef DTYPE_t[:, :] ans_img_th_nwg = ans_img_th_nwg_
    cdef DTYPE_t[:, :] ans_img_th_nwg_fattened = modify_line_width_(ans_img_th_nwg_, radius)
    cdef int th, th_index, search_size, search_index

    cdef DTYPE_t[:, :, :] pred_img_th_nwg_del
    cdef cnp.ndarray[DTYPE_t, ndim=3] pred_img_th_nwg_del_
    cdef list pred_img_th_nwg_del_list = []
    cdef cnp.int32_t[:] del_area_list
    cdef int del_area_list_length

    cdef DTYPE_t[:, :, :] pred_img_th_nwg_del_fattened
    cdef cnp.ndarray[DTYPE_t, ndim=3] pred_img_th_nwg_del_fattened_

    cdef cnp.float64_t[:, :] result

    if otsu:
        threshold_list = np.array([cv2.threshold(pred_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]], dtype=DTYPE)
        threshold_list_length = 1
    else:
        threshold_list = extract_threshold_values(pred_img, min_th=min_th, max_th=max_th)[::step_th]
        threshold_list_length = threshold_list.shape[0]
    pred_img_th_nwg = np.empty((threshold_list_length, ROW, COLUMN), dtype=DTYPE)

    for th_index in range(threshold_list_length):
        if verbose:print(f'\rThreshold: {threshold_list[th_index]}', end=' '*10)
        pred_img_th_list.append(cv2.threshold(pred_img, threshold_list[th_index], 255, cv2.THRESH_BINARY)[1])
    pred_img_th = np.stack(pred_img_th_list)

    if verbose:print(f'\nNWG Start', end='')
    for th_index in prange(threshold_list_length, nogil=True):
        pred_img_th_nwg[th_index] = NWG_(pred_img_th[th_index], symmetric=symmetric)
    
    search_size = 0
    for th_index in range(threshold_list_length):
        if verbose:print(f'\rSearch size: {search_size}', end=' '*10)
        search_size += detect_deleted_area_candidates(np.asarray(pred_img_th_nwg[th_index]), min_area, max_area)[::step_area].shape[0]
    if verbose:print(f'\rSearch size: {search_size}')
    result = np.empty((search_size, 8), dtype=np.float64)
    pred_img_th_nwg_del_ = np.empty((search_size, ROW, COLUMN), dtype=np.uint8)
    search_index = 0
    for th_index in range(threshold_list_length):
        del_area_list = detect_deleted_area_candidates(np.asarray(pred_img_th_nwg[th_index, :, :]), min_area, max_area)[::step_area]
        del_area_list_length = del_area_list.shape[0]
        for del_area_index in range(del_area_list_length):
            if verbose:print(f'\rThreshold: {threshold_list[th_index]}, Del area: {del_area_list[del_area_index]}', end=' '*10)
            pred_img_th_nwg_del_[search_index] = small_area_reduction(np.asarray(pred_img_th_nwg[th_index]), del_area_list[del_area_index])
            result[search_index, 0] = threshold_list[th_index]
            result[search_index, 1] = del_area_list[del_area_index]
            search_index += 1
    pred_img_th_nwg_del = pred_img_th_nwg_del_
    if verbose:print(f'\nmax pred_img_th_nwg_del: {np.max(np.asarray(pred_img_th_nwg_del))}, min pred_img_th_nwg_del: {np.min(np.asarray(pred_img_th_nwg_del))}')

    pred_img_th_nwg_del_fattened = np.empty((search_size, ROW, COLUMN), dtype=np.uint8)
    for search_index in range(search_size):
        if verbose:print(f'\rDilate line... Now index: {search_index+1}', end=' '*10)
        pred_img_th_nwg_del_fattened[search_index] = modify_line_width_(pred_img_th_nwg_del[search_index], radius)
    if verbose:print()
    for search_index in prange(search_size, nogil=True):
        result[search_index, 2:8] = thred_eval(pred_img_th_nwg_del[search_index], ans_img_th_nwg, pred_img_th_nwg_del_fattened[search_index], ans_img_th_nwg_fattened)
    return result
###
