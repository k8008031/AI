#coding:utf-8
#from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity
import cv2
import numpy as np
import os
from os.path import join as pjoin
import shutil
import math

#配置数据
class Config:
    def __init__(self):
        pass
    resizeRate = 0.1  #縮放比例
    min_area = 5000   
    min_contours = 8
    threshold_thresh = 50




#-----------------------------------------------------------------------------------------------
def dealwith(image):
    #获取原始图像的大小 高 寬   通道
    srcHeight,srcWidth ,channels = image.shape

    #对原始图像进行缩放
    image= cv2.resize(image,(int(srcWidth*Config.resizeRate),int(srcHeight*Config.resizeRate)))
    #cv2.imshow("image", image)    嗅出縮放的圖

    #转成灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray) 
    
    
    # 中值滤波平滑，消除噪声 消除顆粒感
    # 当图片缩小后，中值滤波的孔径也要相应的缩小，否则会将有效的轮廓擦除
    #binary = cv2.medianBlur(gray,7)  改數值越大越平滑
    binary = cv2.medianBlur(gray,3)
    

    #转换为二值图像 
    ret, binary = cv2.threshold(binary, Config.threshold_thresh, 255, cv2.THRESH_BINARY)
    #显示转换后的二值图像
    #cv2.imshow("binary", binary)

    # 进行2次腐蚀操作（erosion）
    # 腐蚀操作将会腐蚀图像中白色像素，可以将断开的线段连接起来
    binary = cv2.erode (binary, None, iterations = 2)
    #显示腐蚀后的图像
    #cv2.imshow("erode", binary)

    # canny 边缘检测
    edged = cv2.Canny(binary, 0, 180, apertureSize = 3)
    #显示边缘检测的结果
    #cv2.imshow("edged", edged)

    #cv2.waitKey(0)

    # 提取轮廓
    contours,_ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 输出轮廓数目
    #print("the count of contours is  %d \n"%(len(contours)))


    for idx,c in enumerate(contours):

        approx = cv2.approxPolyDP(c,10,True)#多邊逼近(去雜邊

        cv2.drawContours(image,[approx],-1,(255,0,0),-1)

    return image

#批量图片相似度处理函数simliarval，输入：待处理图片文件夹路径；输出：处理后图片文件夹及基于相似性分类结果：D：/picresult/
#将图片路径存入列表‘pathList’
def simliarval(origidir):  # data_dir是图片集路径
    picpath = []
    for i in os.listdir(origidir):    #文件夹包含的文件或文件夹的名字的列表
        image_dir = pjoin(origidir, i)  # 图片的路径
        picpath.append(image_dir)       #图片路径列表picpath

    pathList = list(picpath)
    picnum = len(pathList)  #長度
    
#参数初始化：k分类标签计数；m处理有效和终止标记；simthd相似图片阈值；otherpath非相似图片文件夹；
    k = 1   #資料夾數
    m = 1   

    simthd = 0.80  #越高分越細
    otherpath =  'D:/picresult'+'/'+'others'

    while m:     #取第一张图片开始搜索
        despath =  'D:/picresult'+'/'+str(k)  #str字串
        img1 = cv2.imread(pathList[0])  #讀取圖片

        ###############################################
        img1 = dealwith(img1)
        #cv2.imshow("img1", img1)
        #cv2.waitKey(0)
        ###############################################
        H1 = cv2.calcHist([img1], [0], None, [256],[0,256])  #色彩直方圖應用  影像, 通道, 遮罩, 區間數量, 數值範圍
        #cv2.imshow("calcHist", H1)
        #cv2.waitKey(0)
        H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1) # 对图片进行归一化处理
        #print("normalize",H1)

        j = 1
        T = picnum-j
        while T > 0:   #进行一次搜索
            img2 = cv2.imread(pathList[j])

            ###############################################
            img2 = dealwith(img2)

            ###############################################
            H2= cv2.calcHist([img2], [0], None, [256],[0,256])
            H2= cv2.normalize(H2, H2, 0, 1, cv2.NORM_MINMAX, -1) # 对图片进行归一化处理


            simval = cv2.compareHist(H1,H2,0) #兩圖比較出來的數值
            #print(simval)  秀出比較值

            if simval > simthd:
                if not os.path.isdir(despath):#檢查目錄是否存在
                    os.makedirs(despath) #創新的資料夾
                shutil.move(pathList[j], despath) #移動圖片到指定資料夾
                #newpicset = np.delete(newpicset,j,0)
                del(pathList[j])#刪除原本照片
                picnum = len(pathList)
            else:
                j = j+1
            T = picnum-j

        if os.path.isdir(despath):  #搜到相似照片的第一张照片存入相应文件夹
            shutil.move(pathList[0], despath)
            del(pathList[0])
            k = k+1
        else:
            if not os.path.isdir(otherpath):   #未搜到相似照片的第一张照片放入otherpath文件夹
                    os.makedirs(otherpath)
            shutil.move(pathList[0], otherpath)
            del(pathList[0])

        picnum = len(pathList)
        print(pathList)
        print(picnum)


        #所有照片搜索完成判断
        if picnum < 1:
            m = 0
origidir = "D:/blured"     #输入图片文件夹
simliarval(origidir)