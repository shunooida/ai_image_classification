import random
import os
import pandas as pd

import process.top as top
import process.download as download
import process.main as main

#ここで作成したzipファイルの名前を終始使用
random_num = random.random()
zipfile_name = str(random_num)

main.main(zipfile_name)

#分類結果が記録されているcsvを読み込む
df = pd.read_csv('csv/' + zipfile_name + '.csv', header=0)
print(df)

df['path'] = './temp/' + zipfile_name + '/' + df['fnames']
print(df)

df['new_path'] = './class/' + zipfile_name + '/' + df['labels'].astype(str) + '/' + df['fnames']
print(df)

for path, new_path in zip(df['path'], df['new_path']):
    os.rename(path, new_path)