import os
import shutil

import pandas as pd

#クラス分け用フォルダ作成(車の分類用)
# def make_folders(zipfile_name, f_num):
#     os.mkdir('class/' + zipfile_name)
#     for i in range(1, f_num+1):
#         os.mkdir('./class/' + zipfile_name + '/' + str(i))
#     return

#新幹線分類用フォルダ作成
def make_shinkansen_folders(zipfile_name):
    os.mkdir('./class/' + zipfile_name)
    os.mkdir('./class/' + zipfile_name + '/700')
    os.mkdir('./class/' + zipfile_name + '/n700a')
    os.mkdir('./class/' + zipfile_name + '/n700s')
    return

#現在のpathをcsvに保存
def add_old_path(df, zipfile_name):
    df['path'] = './temp/' + zipfile_name + '/' + df['fnames']
    return df

#移動先のpathをcsvに保存
def add_new_path(df, zipfile_name):
    #割り振られた番号に従って、フォルダの移動先のパスを作成する
    df['new_path'] = './class/' + zipfile_name + '/' + df['labels'].astype(str) + '/' + df['fnames']
    return df

#zip作成
def make_zip(zipfile_name):
    shutil.make_archive('./download/' + zipfile_name, 'zip', root_dir='./class/' + zipfile_name)