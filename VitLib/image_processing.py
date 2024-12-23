"""画像の処理系をまとめたモジュール"""
import cv2
import numpy as np

def gamma_correction(img:np.ndarray, gamma:float=2.2) -> np.ndarray:
    """ガンマ補正を行う関数

    Args:
        img (numpy.ndarray): 入力画像, 画素値は[0, 255]閉区間の整数値
        gamma (float): ガンマ値

    Returns:
        numpy.ndarray: ガンマ補正された画像
    """
    return (np.power(img/255, 1/gamma)*255).astype(np.uint8)

def change_hue(img:np.ndarray, hue_degree:int) -> np.ndarray:
    """
    画像リスト内の各画像の色相を変更します。

    Args:
        img_list (list): 画像リスト
        hue_degree (int): 色相の変更量(0~180)

    Returns:
        numpy.ndarray: 色相が変更された画像
    """
    if len(img.shape)==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:,:,0] = (img[:,:,0]+hue_degree)%180
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img

def random_hue(img_list:list, hue_degree_range:tuple) -> list:
    """
    画像リスト内の各画像の色相をランダムに変更します。

    Args:
        img_list (list): 画像リスト
        hue_degree_range (tuple): 色相の変更範囲(0~180)

    Returns:
        list: 色相が変更された画像リスト
    """
    hue_degree = np.random.randint(hue_degree_range[0], hue_degree_range[1])
    return [change_hue(img, hue_degree) for img in img_list]

def change_saturation(img:np.ndarray, saturation_ratio:float) -> np.ndarray:
    """
    画像の彩度を変更します。

    Args:
        img (numpy.ndarray): 画像
        saturation_ratio (float): 彩度の変更量(0~1)

    Returns:
        numpy.ndarray: 彩度が変更された画像
    """
    if len(img.shape)==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:,:,1] = np.clip(img[:,:,1]*saturation_ratio, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img

def random_saturation(img_list:list, saturation_ratio_range:tuple) -> list:
    """
    画像リスト内の各画像の彩度をランダムに変更します。

    Args:
        img_list (list): 画像リスト
        saturation_ratio_range (tuple): 彩度の変更範囲(0~1)

    Returns:
        list: 彩度が変更された画像リスト
    """
    saturation_ratio = np.random.uniform(saturation_ratio_range[0], saturation_ratio_range[1])
    return [change_saturation(img, saturation_ratio) for img in img_list]

def change_value(img:np.ndarray, value_ratio:float) -> np.ndarray:
    """
    画像の明度を変更します。

    Args:
        img (numpy.ndarray): 画像
        value_ratio (float): 明度の変更量(0~1)

    Returns:
        numpy.ndarray: 明度が変更された画像
    """
    if len(img.shape)==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:,:,2] = np.clip(img[:,:,2]*value_ratio, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img

def random_value(img_list:list, value_ratio_range:tuple) -> list:
    """
    画像リスト内の各画像の明度をランダムに変更します。

    Args:
        img_list (list): 画像リスト
        value_ratio_range (tuple): 明度の変更範囲(0~1)

    Returns:
        list: 明度が変更された画像リスト
    """
    value_ratio = np.random.uniform(value_ratio_range[0], value_ratio_range[1])
    return [change_value(img, value_ratio) for img in img_list]

def change_contrast(img:np.ndarray, contrast_ratio:float) -> np.ndarray:
    """
    画像のコントラストを変更します。

    Args:
        img (numpy.ndarray): 画像
        contrast_ratio (float): コントラストの変更量(0~1)

    Returns:
        numpy.ndarray: コントラストが変更された画像
    """
    if len(img.shape)==3:
        return cv2.convertScaleAbs(img, alpha=contrast_ratio)
    else:
        return img

def random_contrast(img_list:list, contrast_ratio_range:tuple) -> list:
    """
    画像リスト内の各画像のコントラストをランダムに変更します。
    
    Args:
        img_list (list): 画像リスト
        contrast_ratio_range (tuple): コントラストの変更範囲(0~1)

    Returns:
        list: コントラストが変更された画像リスト
    """
    contrast_ratio = np.random.uniform(contrast_ratio_range[0], contrast_ratio_range[1])
    return [change_contrast(img, contrast_ratio) for img in img_list]

def cut_image(img:np.ndarray, top_left:tuple, size:tuple) -> np.ndarray:
    """
    画像を切り取ります。

    Args:
        img (numpy.ndarray): 画像
        top_left (tuple): 切り取りの左上の座標
        size (tuple): 切り取りのサイズ

    Returns:
        numpy.ndarray: 切り取られた画像
    """
    return img[top_left[0]:top_left[0]+size[0], top_left[1]:top_left[1]+size[1]]

def random_cut_image(img_list:list, size:tuple) -> list:
    """
    画像リスト内の各画像をランダムに切り取ります。
    
    Args:
        img_list (list): 画像リスト
        size (tuple): 切り取りのサイズ
    
    Returns:
        list: 切り取られた画像リスト
    """
    img_size = img_list[0].shape
    top_left = (np.random.randint(0, img_size[0]-size[0]), np.random.randint(0, img_size[1]-size[1]))
    return [cut_image(img, top_left, size) for img in img_list]

def select_cut_image(img_list:list, top_left:tuple, size:tuple) -> list:
    """
    画像リスト内の各画像を指定した位置とサイズで切り取ります。

    Args:
        img_list (list): 画像リスト
        top_left (tuple): 切り取りの左上の座標
        size (tuple): 切り取りのサイズ
    
    Returns:
        list: 切り取られた画像リスト
    """
    return [cut_image(img, top_left, size) for img in img_list]

def rotate_image(img:np.ndarray, rotate_times:int) -> np.ndarray:
    """
    画像を回転します。(1回転につき90度)

    Args:
        img (numpy.ndarray): 画像
        rotate_times (int): 回転回数(1回につき90度)

    Returns:
        numpy.ndarray: 回転された画像
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
    """
    画像リスト内の各画像をランダムに回転します。

    Args:
        img_list (list): 画像リスト

    Returns:
        list: 回転された画像リスト
    """
    rotate_times = np.random.randint(0, 3)
    return [rotate_image(img, rotate_times) for img in img_list]

def flip_image(img:np.ndarray, flip_code:int) -> np.ndarray:
    """
    画像を反転します。

    Args:
        img (numpy.ndarray): 画像
        flip_code (int): 反転コード
            - 0: 上下反転
            - 1: 左右反転
            - -1: 上下左右反転

    Returns:
        numpy.ndarray: 反転された画像
    """
    return cv2.flip(img, flip_code)

def random_flip_image(img_list:list) -> list:
    """
    画像リスト内の各画像をランダムに反転します。

    Args:
        img_list (list): 画像リスト
        flip_code_range (tuple): 反転コードの範囲
            - 0: 上下反転
            - 1: 左右反転
            - -1: 上下左右反転
            - 2: なし

    Returns:
        list: 反転された画像リスト
    """
    flip_code = np.random.randint(-1, 2)
    if flip_code==2:
        return img_list
    return [flip_image(img, flip_code) for img in img_list]

def img_show(img:np.ndarray):
    """
    画像を表示します。

    Args:
        img (numpy.ndarray): 画像
    """
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
