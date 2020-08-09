import os
import re
import numpy as np
import time
import glob
import shutil
import PIL.Image
from PIL import ImageEnhance
import subprocess
import cv2

#uer_name = input("Who are you")
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#----------------------------------------------
#mov, pic, posをクリアする

os.chdir('cascade')
for x in glob.glob('*.xml'):
    os.remove(x)

os.chdir('../')
os.chdir('pic')
for x in glob.glob('*.jpg'):
    os.remove(x)

os.chdir('../')
os.chdir('pos')
for x in glob.glob('*.jpg'):
    os.remove(x)

os.chdir('../')
os.remove('pos.txt')
#----------------------------------------------
#pos.txtを作成
f = open('pos.txt', 'a')

#----------------------------------------------
#顔の画像を集める
cap = cv2.VideoCapture(0)
cascade_path1 = "haarcascade_frontalface_default.xml"
cascade_path2 = 'lbpcascade_profileface.xml'
cascade1 = cv2.CascadeClassifier(cascade_path1)
cascade2 = cv2.CascadeClassifier(cascade_path2)
color = (255,255,255)
picture_num = 1
while True:
    ret, frame = cap.read()
    facerect1 = cascade1.detectMultiScale(frame, scaleFactor=1.7, minNeighbors=4, minSize=(100,100))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, str(picture_num), (10,500), font, 4,(0,0,0),2,cv2.LINE_AA)
    if len(facerect1) > 0:
        for (x,y,w,h) in facerect1:
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            picture_name = 'pic/pic' + str(picture_num) + '.jpg'
            cv2.imwrite(picture_name, frame)
            #text = picture_name + ' 1 ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n'
            #f.write(text)
            picture_num = picture_num + 1
    cv2.imshow("frame", frame)
    if picture_num == 41:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

#----------------------------------------------
#水増し開始
#picturesの写真の数を数える
dir = os.getcwd()
dirPic = dir + "/pic"
files = os.listdir(dirPic)
count = 0
for file in files:
    count = count + 1
os.chdir('pic')
#写真の枚数
imageNum = count
#左右反転
for num in range(1, count+1):
    name = 'pic' + str(num) + '.jpg'
    if os.path.exists(name) :
        pass
    else :
        continue
    if os.path.getsize(name) == 0:
        os.remove(name)
        continue
    img = cv2.imread(name)
    yAxis = cv2.flip(img, 1)
    newName = 'pic' + str(imageNum) + '.jpg'
    cv2.imwrite(newName,yAxis)
    imageNum = imageNum + 1
print('OK')
print('NEXT STAGE')

#写真の数をカウント
dir = os.getcwd()
files = os.listdir(dir)
count = 0
for file in files:
    count = count + 1
print(count)

#写真の枚数
imageNum = count
#imageNumを固定しておく
picNum = imageNum


SATURATION = 0.5
CONTRAST = 0.5
BRIGHTNESS = 0.5
SHARPNESS = 2.0
# 彩度を変える
for num in range(1, count+1):
    name = 'pic' + str(num) + '.jpg'
    if os.path.exists(name) :
        pass
    else :
        print('NO')
        continue
    if os.path.getsize(name) == 0:
        os.remove(name)
        continue
    img = PIL.Image.open(name)
    saturation_converter = ImageEnhance.Color(img)
    saturation_img = saturation_converter.enhance(SATURATION)
    newName = 'pic' + str(imageNum) + '.jpg'
    saturation_img.save(newName)
    imageNum = imageNum + 1
print('OK')
print('NEXT STAGE')

#写真の数をカウント
dir = os.getcwd()
files = os.listdir(dir)
count = 0
for file in files:
    count = count + 1
print(count)

#写真の枚数
imageNum = count

count = picNum
# コントラストを変える
for num in range(1, count+1):
    name = 'pic' + str(num) + '.jpg'
    if os.path.exists(name) :
        pass
    else :
        print('NO')
        continue
    if os.path.getsize(name) == 0:
        os.remove(name)
        continue
    img = PIL.Image.open(name)
    contrast_converter = ImageEnhance.Contrast(img)
    contrast_img = contrast_converter.enhance(CONTRAST)
    newName = 'pic' + str(imageNum) + '.jpg'
    contrast_img.save(newName)
    imageNum = imageNum + 1
print('OK')
print('NEXT STAGE')

#写真の数をカウント
dir = os.getcwd()
files = os.listdir(dir)
count = 0
for file in files:
    count = count + 1
print(count)

#写真の枚数
imageNum = count

count = picNum
# 明度を変える
for num in range(1, count+1):
    name = 'pic' + str(num) + '.jpg'
    if os.path.exists(name) :
        pass
    else :
        print('NO')
        continue
    if os.path.getsize(name) == 0:
        os.remove(name)
        continue
    img = PIL.Image.open(name)
    brightness_converter = ImageEnhance.Brightness(img)
    brightness_img = brightness_converter.enhance(BRIGHTNESS)
    newName = 'pic' + str(imageNum) + '.jpg'
    brightness_img.save(newName)
    imageNum = imageNum + 1
print('OK')
print('NEXT STAGE')

#写真の数をカウント
dir = os.getcwd()
files = os.listdir(dir)
count = 0
for file in files:
    count = count + 1
print(count)

#写真の枚数
imageNum = count

count = picNum
# シャープネスを変える
for num in range(1, count+1):
    name = 'pic' + str(num) + '.jpg'
    if os.path.exists(name) :
        pass
    else :
        print('NO')
        continue
    if os.path.getsize(name) == 0:
        os.remove(name)
        continue
    img = PIL.Image.open(name)
    sharpness_converter = ImageEnhance.Sharpness(img)
    sharpness_img = sharpness_converter.enhance(SHARPNESS)
    newName = 'pic' + str(imageNum) + '.jpg'
    sharpness_img.save(newName)
    imageNum = imageNum + 1
print('OK')
print('NEXT STAGE')

#写真の数をカウント
dir = os.getcwd()
files = os.listdir(dir)
count = 0
for file in files:
    count = count + 1
print(count)

#写真の枚数
imageNum = count
#imageNumを固定しておく
picNum = imageNum

#15度回転
for num in range(1, count+1):
    name = 'pic' + str(num) + '.jpg'
    if os.path.exists(name) :
        pass
    else :
        print('NO')
        continue
    if os.path.getsize(name) == 0:
        os.remove(name)
        continue
    # 画像読み込み
    img = cv2.imread(name)
    h, w = img.shape[:2]
    size = (w, h)

    # 回転角の指定
    angle = 15
    angle_rad = angle/180.0*np.pi

    # 回転後の画像サイズを計算
    w_rot = int(np.round(h*np.absolute(np.sin(angle_rad))+w*np.absolute(np.cos(angle_rad))))
    h_rot = int(np.round(h*np.absolute(np.cos(angle_rad))+w*np.absolute(np.sin(angle_rad))))
    size_rot = (w_rot, h_rot)

    # 元画像の中心を軸に回転する
    center = (w/2, h/2)
    scale = 1.0
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # 平行移動を加える (rotation + translation)
    affine_matrix = rotation_matrix.copy()
    affine_matrix[0][2] = affine_matrix[0][2] -w/2 + w_rot/2
    affine_matrix[1][2] = affine_matrix[1][2] -h/2 + h_rot/2

    img_rot = cv2.warpAffine(img, affine_matrix, size_rot, flags=cv2.INTER_CUBIC)

    cv2.imwrite(newName, img_rot)
    newName = 'pic' + str(imageNum) + '.jpg'
    saturation_img.save(newName)
    imageNum = imageNum + 1
print('OK')
print('NEXT STAGE')

#写真の数をカウント
dir = os.getcwd()
files = os.listdir(dir)
count = 0
for file in files:
    count = count + 1
print(count)

#写真の枚数
imageNum = count

#-15度回転
for num in range(1, count+1):
    name = 'pic' + str(num) + '.jpg'
    if os.path.exists(name) :
        pass
    else :
        print('NO')
        continue
    if os.path.getsize(name) == 0:
        os.remove(name)
        continue
    # 画像読み込み
    img = cv2.imread(name)
    h, w = img.shape[:2]
    size = (w, h)

    # 回転角の指定
    angle = -15
    angle_rad = angle/180.0*np.pi

    # 回転後の画像サイズを計算
    w_rot = int(np.round(h*np.absolute(np.sin(angle_rad))+w*np.absolute(np.cos(angle_rad))))
    h_rot = int(np.round(h*np.absolute(np.cos(angle_rad))+w*np.absolute(np.sin(angle_rad))))
    size_rot = (w_rot, h_rot)

    # 元画像の中心を軸に回転する
    center = (w/2, h/2)
    scale = 1.0
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # 平行移動を加える (rotation + translation)
    affine_matrix = rotation_matrix.copy()
    affine_matrix[0][2] = affine_matrix[0][2] -w/2 + w_rot/2
    affine_matrix[1][2] = affine_matrix[1][2] -h/2 + h_rot/2

    img_rot = cv2.warpAffine(img, affine_matrix, size_rot, flags=cv2.INTER_CUBIC)

    cv2.imwrite(newName, img_rot)
    newName = 'pic' + str(imageNum) + '.jpg'
    saturation_img.save(newName)
    imageNum = imageNum + 1
print('OK')
print('NEXT STAGE')

#写真の数をカウント
dir = os.getcwd()
files = os.listdir(dir)
count = 0
for file in files:
    count = count + 1
print(count)
print('OK')
print('COMPLETE')

#------------------------------------------------------
#テキストファイル作成
#cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#os.chdir('pic')
for num in glob.glob('*.jpg'):
    img = cv2.imread(num)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        text = 'pic/' + num + ' 1 ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n'
        f.write(text)

cmd = 'opencv_createsamples -info pos.txt -vec pos.vec -num ' + str(count)
print(cmd)
#subprocess.call(cmd)
#os.system('opencv_createsamples -info pos.txt -vec pos.vec -num ' + str(count))

#print('OK')

cmd = 'opencv_traincascade -data ./cascade -vec pos.vec -bg neg.txt -numPos 1500 numNeg 255'
print(cmd)

#os.system( cmd )
#subprocess.call(cmd)
#os.rename('cascade/cascade.xml', 'cascade_' + uer_name + '.xml')
#shutil.move('cascade_' + uer_name + '.xml', 'casacade_file')
print('COMPLETE.')