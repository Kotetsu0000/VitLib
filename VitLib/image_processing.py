"""画像の処理系をまとめたモジュール

Todo:
    - 画像の色相を変更する関数を追加する
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

def gamma_correction(img:np.ndarray, gamma:float=2.2):
    """ガンマ補正を行う関数

    Args:
        img (numpy.ndarray): 入力画像, 画素値は[0, 255]閉区間の整数値
        gamma (float): ガンマ値

    Returns:
        numpy.ndarray: ガンマ補正された画像
    """
    return (np.power(img/255, 1/gamma)*255).astype(np.uint8)

def change_hue(img_list:list, hue_degree:float):
    """
    画像リスト内の各画像の色相を変更します。

    Args:
        img_list (list): 画像リスト
        hue_degree (float): 色相の変更量

    Returns:
        list: 色相が変更された画像リスト
    """
    for i in range(len(img_list)):
        img_list[i] = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2HSV)
        img_list[i][:,:,0] = (img_list[i][:,:,0]+hue_degree)%180
        img_list[i] = cv2.cvtColor(img_list[i], cv2.COLOR_HSV2BGR)
    return img_list
