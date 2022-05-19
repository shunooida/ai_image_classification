import os
import shutil
import zipfile

def make_dir(zipfile_name):
    path = './temp/' + zipfile_name
    os.mkdir(path)

def move_dir(zipfile_name):
    from_path = './receive/' + zipfile_name + '.zip'
    to_path = './temp/' + zipfile_name + '/' + zipfile_name + '.zip'
    shutil.move(from_path, to_path)

# def open_zip(zipfile_name):
#     with zipfile.ZipFile('./temp/' + zipfile_name + '/' + zipfile_name + '.zip') as existing_zip:
#         existing_zip.extractall('./temp/' + zipfile_name + '/')

def open_zip(zipfile_name):
    zp = zipfile.ZipFile('./temp/' + zipfile_name + '/' + zipfile_name + '.zip', "r")
    zp.extractall(path='./temp/' + zipfile_name + '/')
    zp.close()

def remove_zip(zipfile_name):
    os.remove('./temp/' + zipfile_name + '/' + zipfile_name + '.zip')
    #os.remove('afterzip/test.zip')