import re
import shutil
import zipfile
from flask import Flask
from flask import request
from flask import render_template
from flask import make_response
from werkzeug.utils import secure_filename
import random
import os
import pandas as pd

import process.top as top
import process.download as download
import process.main as main

random_num = random.random()

#ここで作成したzipファイルの名前を終始使用
zipfile_name = str(random_num)

#Flaskオブジェクト作成
app = Flask(__name__)

@app.route('/')
@app.route('/top')
@app.route('/download', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':

        # #分類結果が記録されているcsvを読み込む
        # df = pd.read_csv('csv/' + zipfile_name + '.csv', header=0)
        
        f = request.files["the_file"]
        #任意の階層を相対パスで指定
        f.save('./receive/' + zipfile_name + '.zip')
        #f.save('./receive/' + secure_filename(f.filename))

        #tempフォルダの中に新規フォルダを作成
        #ここでパスは変更しなくてはいけない
        top.make_dir(zipfile_name)
        top.move_dir(zipfile_name)
        top.open_zip(zipfile_name)
        top.remove_zip(zipfile_name)

        #モデル実行
        main.main(zipfile_name)

        download.make_shinkansen_folders(zipfile_name)
        
        #分類結果が記録されているcsvを読み込む
        df = pd.read_csv('csv/' + zipfile_name + '.csv', header=0)

        df = download.add_old_path(df, zipfile_name)
        df = download.add_new_path(df, zipfile_name)

        #分類結果の取得
        total_sum = df['labels'].count()
        sum_n700a = df[df['labels']=='n700a'].count()[1]
        sum_n700s = df[df['labels']=='n700s'].count()[1]
        sum_700 = df[df['labels']=='700'].count()[1]
        

        # 古いパスから新しいパスに更新することで、フォルダを移動する
        for path, new_path in zip(df['path'], df['new_path']):
            os.rename(path, new_path)
        download.make_zip(zipfile_name)
        
        #不要ディレクトリ削除
        shutil.rmtree('class/' + zipfile_name + '/')
        shutil.rmtree('temp/' + zipfile_name + '/')

        #不要CSV削除
        os.remove('csv/' + zipfile_name + '.csv')

        #アップロードしてサーバーにファイルが保存されたらdownload.htmlを表示
        return render_template('download.html', total_sum = total_sum, sum_n700a = sum_n700a, sum_n700s = sum_n700s, sum_700 = sum_700)
    else:
    	#GETでアクセスされた時、top.htmlを表示
    	return render_template('top.html')

@app.route("/downloadzip")
def downloadzip():
    response = make_response()
    response.data  = open('download/' + zipfile_name + '.zip', "rb").read()
    response.headers['Content-Type'] = 'application/octet-stream'
    response.headers['Content-Disposition'] = 'attachment; filename=download.zip'
    return response

@app.route("/back")
def back_to_top():
    os.remove('download/' + zipfile_name + '.zip')
    return render_template('top.html')

#以下必須記述
if __name__ == '__main__':
    app.run()