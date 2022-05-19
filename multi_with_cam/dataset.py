import os
from glob import glob
import re 

import numpy as np
import pandas as pd
import scipy.io

from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report
from sklearn import preprocessing
import datetime


path_input = 'data\cars\car_devkit\devkit\cars_train_annos.mat'
path_output = 'data\cars\car_devkit\devkit\cars_train_labels.csv'



#
#
#
def read_mat_to_label(path_input, path_output, mode='train'):
    """
    Stanford cars データセットのmatファイルを読み込み、
    ファイル名とラベルのみをcsvファイルで出力する。
    trainのときは、mode='train'
    testのときは、mode='test'
    
    例えば、pathはこんな感じ
    path_input = '../data/cars\\car_devkit/devkit\\cars_train_annos.mat'
    path_output = '../data/cars\\car_devkit/devkit\\cars_train_labels.csv'
    """
    annos = scipy.io.loadmat(path_input)['annotations'][0]
    fnames, labels = [], []
    
    if mode == 'train':
        for i, anno in enumerate(annos):
            label = anno[4][0][0]
            fname = anno[5][0]
            labels.append(label)
            fnames.append(fname)
            
    elif mode == 'test':
        for i, anno in enumerate(annos):
            fname = anno[4][0]
            fnames.append(fname)
        labels = [0] * len(fnames)  # test 時は、すべて0のラベルをつける

    else:
        print('modeが適切ではありません。')
        return
    
    df = pd.DataFrame({'fnames': fnames, 'labels': labels})
    df.to_csv(path_output, index=False, encoding='utf-8')
    return None


def get_fnames_and_labels(path, fname_col='fnames', label_col='labels'):
    """
     csvファイルを読み取り、dataloader を作成するために、\n
     fnames, labelsを返す。
     csvファイルは、
     カラム名は {'fnames', 'labels'} をデフォルトの引数にする。
    """
    df = pd.read_csv(path)
    fnames = df[fname_col].values  # transforms.Compose()には、ndarray型を入れる
    labels = df[label_col].to_list()  # LabelEncoder()は、リスト型を入れる
    return fnames, labels


# ここを参照 -> https://www.nogawanogawa.com/entry/resnet_pytorch
class MyDataSet(Dataset):
    """
    オリジナルのデータセットを作成するためのクラス。\n
    受け取ったDatasetは、\n
    torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) \n
    を用いて DataLoader にする。
    Attributes
    ----------
    fnames : list
        file name が列挙されたリスト
    labels : list
        file name が属するクラスを列挙したリスト
        fnames とインデックスが対応するようにしてください。
    paht : path
        画像ファイルが保存されているディレクトリ。
        例） path = './images/'
    """
    
    def __init__(self, fnames, labels, path):
        
        self.images = [path + fname for fname in fnames]
        self.labels = labels
        self.le = preprocessing.LabelEncoder()

        self.le.fit(self.labels)
        self.labels_id = self.le.transform(self.labels)
        
        # 画像は224 x 224 に圧縮
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image = image.convert('RGB')
        label = self.labels_id[idx]
        return self.transform(image), int(label)

read_mat_to_label(path_input, path_output, mode='train')