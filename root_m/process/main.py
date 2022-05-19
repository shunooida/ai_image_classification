import os
from glob import glob
import re
import sys

import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import torchvision.models
import torch.nn as nn

# 手作りのモジュール
import process.dataset as dataset # 必ずインポートしといてください

import process.multiscale as multiscale

# -------------------------------------------------------------------------------------------
#
# main.py に記述する処理
#
def main(csv_file_name):
    global device

    imgs_dir = './temp/' + csv_file_name

    # モデルの呼び出し（デバッグ用）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnet18 = load_trained_resnet18(model_path='./model/multiscale.pth')

    # modelで画像を分類してCSVにする処理
    classify(resnet18, imgs_dir, csv_file_name)
    print('done !')
# -----------------------------------------------------------------------------------------    
    
#
# モデルを呼び出す関数  ※デバッグ用
#
def load_trained_resnet18(model_path):
    
    multi_model = multiscale.Multi()
    
    # model_path が合致すればロードする
    # try:
    # weight = model['fc.weight']
    # print(weight.size())
    #resnet18.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    multi_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print('学習済みパラメータ適用完了')
    # except:
    #     pass
    
    # Get cpu or gpu device for training.
    multi_model.to(device)
    return multi_model

#
# クラス分類を行い、csv を吐き出す
#
def classify(model, imgs_dir_path, csv_file_name):
    
    def predict(dataloader, model):
        y_pred = []
        with torch.no_grad():
            for X, _ in dataloader:
                X = X.to(device)
                pred = model(X)
                y_pred += [int(l.argmax()) for l in pred]
        return y_pred
    
    # dataset, dataloader作成
    test_fnames = [path.split('\\')[-1] for path in glob(imgs_dir_path+'/*')]
    testset = dataset.MyDataSet(
        test_fnames, 
        labels=[0]*len(test_fnames),  # dataset クラスは、labels が必要なので、仮のラベルとして0をおく
        path=imgs_dir_path + '/')
    test_loader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False)
    
    # クラス分類
    y_pred = predict(test_loader, model)
    
    # pd.DataFrame 作成、csv 出力処理
    label_dict = {
        0: '700',
        1: 'n700a',
        2: 'n700s'}
    csv_path='./csv/{}.csv'.format(csv_file_name)
    pd.DataFrame(
        {'fnames': test_fnames,
         'labels': [label_dict[p] for p in y_pred]}
    ).to_csv(csv_path, index=False, encoding='utf-8')

    return 
