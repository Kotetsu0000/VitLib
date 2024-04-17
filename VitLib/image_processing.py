"""画像の処理系をまとめたモジュール

Todo:
    - 画像の明るさを変更する関数を追加する
    - 画像の彩度を変更する関数を追加する
    - 画像のコントラストを変更する関数を追加する
    - 画像の切り取りを行う関数を追加する
    - 画像の回転の関数を追加する
    - 画像の上下左右反転の関数を追加する
    - 画像を表示する関数を追加する
"""
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
        list: 色相が変更された画像リスト
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img[:,:,0] = (img[:,:,0]+hue_degree)%180
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img

def change_saturation(img:np.ndarray, saturation_ratio:float) -> np.ndarray:
    """
    画像の彩度を変更します。

    Args:
        img (numpy.ndarray): 画像
        saturation_ratio (float): 彩度の変更量(0~1)

    Returns:
        numpy.ndarray: 彩度が変更された画像
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img[:,:,1] = np.clip(img[:,:,1]*saturation_ratio, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img

def img_show(img:np.ndarray):
    """
    画像を表示します。

    Args:
        img (numpy.ndarray): 画像
    """
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
