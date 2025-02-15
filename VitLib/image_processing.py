import cv2
import numpy as np

def gamma_correction(img: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """画像にガンマ補正を適用する関数

    Args:
        img (np.ndarray): 画像 (画素値: 0〜255)
        gamma (float): ガンマ値

    Returns:
        np.ndarray: ガンマ補正後の画像

    Example:
        >>> import cv2
        >>> from VitLib import gamma_correction
        >>> img = cv2.imread('example.jpg')
        >>> gamma_correction(img, 2.2)

    Note:
        画像は浮動小数点数に変換後、補正し再スケールされます。
    """
    return (np.power(img/255, 1/gamma)*255).astype(np.uint8)

def change_hue(img:np.ndarray, hue_degree:int) -> np.ndarray:
    """画像のリスト内の各画像の色相を変更する関数

    Args:
        img_list (list): 画像のリスト
        hue_degree (int): 色相の変更量(0~180)

    Returns:
        np.ndarray: 色相が変更された画像

    Example:
        >>> import cv2
        >>> from VitLib import change_hue
        >>> img = cv2.imread('example.jpg')
        >>> change_hue(img, 30)
    """
    if len(img.shape)==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:,:,0] = (img[:,:,0]+hue_degree)%180
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img

def random_hue(img_list:list, hue_degree_range:tuple) -> list:
    """画像のリスト内の各画像の色相をランダムに変更する関数

    Args:
        img_list (list): 画像のリスト
        hue_degree_range (tuple): 色相の変更範囲(0~180)

    Returns:
        list: 色相が変更された画像のリスト

    Example:
        >>> import cv2
        >>> from VitLib import random_hue
        >>> img = cv2.imread('example.jpg')
        >>> random_hue([img], (0, 180))
    """
    hue_degree = np.random.randint(hue_degree_range[0], hue_degree_range[1])
    return [change_hue(img, hue_degree) for img in img_list]

def change_saturation(img:np.ndarray, saturation_ratio:float) -> np.ndarray:
    """画像の彩度を変更する関数

    Args:
        img (np.ndarray): 画像 (画素値: 0〜255)
        saturation_ratio (float): 彩度の変更量(0~1)

    Returns:
        np.ndarray: 彩度が変更された画像

    Example:
        >>> import cv2
        >>> from VitLib import change_saturation
        >>> img = cv2.imread('example.jpg')
        >>> change_saturation(img, 0.5)
    """
    if len(img.shape)==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:,:,1] = np.clip(img[:,:,1]*saturation_ratio, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img

def random_saturation(img_list: list, saturation_ratio_range: tuple) -> list:
    """画像のリスト内の各画像の彩度をランダムに変更する関数

    Args:
        img_list (list): 画像のリスト
        saturation_ratio_range (tuple): 彩度の変更範囲(0~1)

    Returns:
        list: 彩度が変更された画像のリスト

    Example:
        >>> import cv2
        >>> from VitLib import random_saturation
        >>> img = cv2.imread('example.jpg')
        >>> random_saturation([img], (0, 1))
    """
    saturation_ratio = np.random.uniform(saturation_ratio_range[0], saturation_ratio_range[1])
    return [change_saturation(img, saturation_ratio) for img in img_list]

def change_value(img:np.ndarray, value_ratio:float) -> np.ndarray:
    """画像の明度を変更する関数

    Args:
        img (np.ndarray): 画像 (画素値: 0〜255)
        value_ratio (float): 明度の変更量(0~1)

    Returns:
        np.ndarray: 明度が変更された画像

    Example:
        >>> import cv2
        >>> from VitLib import change_value
        >>> img = cv2.imread('example.jpg')
        >>> change_value(img, 0.5)
    """
    if len(img.shape)==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:,:,2] = np.clip(img[:,:,2]*value_ratio, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img

def random_value(img_list:list, value_ratio_range:tuple) -> list:
    """画像のリスト内の各画像の明度をランダムに変更します。

    Args:
        img_list (list): 画像のリスト
        value_ratio_range (tuple): 明度の変更範囲(0~1)

    Returns:
        list: 明度が変更された画像のリスト

    Example:
        >>> import cv2
        >>> from VitLib import random_value
        >>> img = cv2.imread('example.jpg')
        >>> random_value([img], (0, 1))
    """
    value_ratio = np.random.uniform(value_ratio_range[0], value_ratio_range[1])
    return [change_value(img, value_ratio) for img in img_list]

def change_contrast(img:np.ndarray, contrast_ratio:float) -> np.ndarray:
    """画像のコントラストを変更する関数

    Args:
        img (np.ndarray): 画像 (画素値: 0〜255)
        contrast_ratio (float): コントラストの変更量(0~1)

    Returns:
        np.ndarray: コントラストが変更された画像

    Example:
        >>> import cv2
        >>> from VitLib import change_contrast
        >>> img = cv2.imread('example.jpg')
        >>> change_contrast(img, 0.5)
    """
    if len(img.shape)==3:
        return cv2.convertScaleAbs(img, alpha=contrast_ratio)
    else:
        return img

def random_contrast(img_list: list, contrast_ratio_range: tuple) -> list:
    """画像のリスト内の各画像のコントラストをランダムに変更する関数

    Args:
        img_list (list): 画像のリスト
        contrast_ratio_range (tuple): コントラストの変更範囲(0~1)

    Returns:
        list: コントラストが変更された画像のリスト

    Example:
        >>> import cv2
        >>> from VitLib import random_contrast
        >>> img = cv2.imread('example.jpg')
        >>> random_contrast([img], (0, 1))
    """
    contrast_ratio = np.random.uniform(contrast_ratio_range[0], contrast_ratio_range[1])
    return [change_contrast(img, contrast_ratio) for img in img_list]

def cut_image(img: np.ndarray, top_left: tuple, size: tuple) -> np.ndarray:
    """画像を指定された位置とサイズで切り取る関数

    Args:
        img (np.ndarray): 入力画像
        top_left (tuple): 切り取り開始位置 (x, y)
        size (tuple): 切り取りサイズ (幅, 高さ)

    Returns:
        np.ndarray: 切り取られた画像

    Example:
        >>> import cv2
        >>> from VitLib import cut_image
        >>> img = cv2.imread('example.jpg')
        >>> cut_image(img, (100, 100), (200, 200))
    """
    return img[top_left[0]:top_left[0]+size[0], top_left[1]:top_left[1]+size[1]]

def random_cut_image(img_list:list, size:tuple) -> list:
    """画像のリスト内の各画像をランダムに切り取る関数
    
    Args:
        img_list (list): 画像のリスト
        size (tuple): 切り取りのサイズ
    
    Returns:
        list: 切り取られた画像のリスト

    Example:
        >>> import cv2
        >>> from VitLib import random_cut_image
        >>> img = cv2.imread('example.jpg')
        >>> random_cut_image([img], (200, 200))
    """
    img_size = img_list[0].shape
    top_left = (np.random.randint(0, img_size[0]-size[0]), np.random.randint(0, img_size[1]-size[1]))
    return [cut_image(img, top_left, size) for img in img_list]

def select_cut_image(img_list:list, top_left:tuple, size:tuple) -> list:
    """画像のリスト内の各画像を指定した位置とサイズで切り取る関数

    Args:
        img_list (list): 画像のリスト
        top_left (tuple): 切り取りの左上の座標
        size (tuple): 切り取りのサイズ
    
    Returns:
        list: 切り取られた画像のリスト

    Example:
        >>> import cv2
        >>> from VitLib import select_cut_image
        >>> img = cv2.imread('example.jpg')
        >>> select_cut_image([img], (100, 100), (200, 200))
    """
    return [cut_image(img, top_left, size) for img in img_list]

def rotate_image(img: np.ndarray, rotate_times: int) -> np.ndarray:
    """画像を回転する関数

    Args:
        img (np.ndarray): 入力画像
        rotate_times (int): 90度単位で回転する回数 (正なら時計回り、負なら反時計回り)

    Returns:
        np.ndarray: 回転後の画像

    Example:
        >>> import cv2
        >>> from VitLib import rotate_image
        >>> img = cv2.imread('example.jpg')
        >>> rotate_image(img, 1)
    """
    if rotate_times%4==0:
        return img
    elif rotate_times%4==1:
        rotate_code = cv2.ROTATE_90_CLOCKWISE
    elif rotate_times%4==2:
        rotate_code = cv2.ROTATE_180
    elif rotate_times%4==3:
        rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE
    return cv2.rotate(img, rotate_code)

def random_rotate_image(img_list:list) -> list:
    """画像のリスト内の各画像をランダムに回転する関数

    Args:
        img_list (list): 画像のリスト

    Returns:
        list: 回転された画像のリスト

    Example:
        >>> import cv2
        >>> from VitLib import random_rotate_image
        >>> img = cv2.imread('example.jpg')
        >>> random_rotate_image([img])
    """
    rotate_times = np.random.randint(0, 3)
    return [rotate_image(img, rotate_times) for img in img_list]

def flip_image(img:np.ndarray, flip_code:int) -> np.ndarray:
    """画像を反転する関数

    Args:
        img (np.ndarray): 画像 (画素値: 0〜255)
        flip_code (int): 反転コード

            - 0: 上下反転
            - 1: 左右反転
            - -1: 上下左右反転

    Returns:
        np.ndarray: 反転された画像

    Example:
        >>> import cv2
        >>> from VitLib import flip_image
        >>> img = cv2.imread('example.jpg')
        >>> flip_image(img, 1)
    """
    return cv2.flip(img, flip_code)

def random_flip_image(img_list:list) -> list:
    """画像のリスト内の各画像をランダムに反転する関数

    Args:
        img_list (list): 画像のリスト
        flip_code_range (tuple): 反転コードの範囲

            - 0: 上下反転
            - 1: 左右反転
            - -1: 上下左右反転
            - 2: なし

    Returns:
        list: 反転された画像のリスト

    Example:
        >>> import cv2
        >>> from VitLib import random_flip_image
        >>> img = cv2.imread('example.jpg')
        >>> random_flip_image([img])
    """
    flip_code = np.random.randint(-1, 2)
    if flip_code==2:
        return img_list
    return [flip_image(img, flip_code) for img in img_list]

def img_show(img: np.ndarray):
    """画像を表示する関数

    Args:
        img (np.ndarray): 表示する画像

    Example:
        >>> import cv2
        >>> from VitLib import img_show
        >>> img = cv2.imread('example.jpg')
        >>> img_show(img)
    """
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
