import argparse

import cv2
import numpy as np
import onnxruntime as ort
# import torch

# from ultralytics.yolo.utils import ROOT, yaml_load
# from ultralytics.yolo.utils.checks import check_requirements, check_yaml
import yaml 
import re
import time
def four_point_transform( image, pts):
        	# obtain a consistent order of the points and unpack them
    	# individually
    	# rect = self.order_points(pts)
        padding = 20

        rect= pts
        (tl, tr, br, bl) = rect
    	# compute the width of the new image, which will be the
    	# maximum distance between bottom-right and bottom-left
    	# x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
        	[0, 0],
        	[maxWidth - 1, 0],
        	[maxWidth - 1, maxHeight - 1],
        	[0, maxHeight - 1]], dtype = "float32")
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        # return the warped image
        return warped
def get_padding_align(pts1, pts2, ratio=0.2):
    a = (pts1[1]-pts2[1])/(pts1[0]-pts2[0])
    b= pts1[1]-a*pts1[0]
    
    if pts1[0]> pts2[0]:
        x1 = pts1[0] + ratio*(pts1[0]-pts2[0])
        x2 = pts2[0] - ratio*(pts1[0]-pts2[0])
    else:
        x2 = pts2[0] + ratio*(pts2[0]-pts1[0])
        x1 = pts1[0] - ratio*(pts2[0]-pts1[0])
        
    y1 = a*x1+b
    y2 = a*x2+b 
    return x1,y1,x2,y2
'''if __name__ == '__main__':
    import copy
    import os
    import pathlib
    import json
    count=1
    data_path='dataset/train_quality/separate_label/old_data/re_align/22h_cam16'
    for fn in os.listdir(data_path):
            if pathlib.Path(fn).suffix != '.json':
                try:
                    extension = pathlib.Path(fn).suffix
                    jsonfile=fn.replace(extension,'.json')
                    file_label_data = json.load(open(data_path+'/'+fn.replace(extension,'.json'),'r'))
                    img=cv2.imread(data_path+'/'+fn)
                    out_path='dataset/train_quality/align'
                    output_filename =  out_path+'/'+fn
                    #print(output_filename)
                    #print((file_label_data['shapes'][0]['points']))
                    yolo_kps = np.array(file_label_data['shapes'][0]['points'])
                    #print(yolo_kps)
                    tmp = copy.deepcopy(yolo_kps[3])
                    yolo_kps[3]= yolo_kps[2]
                    yolo_kps[2]= tmp
                    # print(yolo_kps.shape)
                    yldm = copy.deepcopy(yolo_kps)	
                    #print(yldm)
                    ratio = 0
                    x1,y1,x2,y2 = get_padding_align(yldm[1],yldm[2],ratio)
                    x0,y0,x3,y3 = get_padding_align(yldm[0],yldm[3],ratio)
                    yldm[0]= np.array([x0,y0])
                    yldm[1]= np.array([x1,y1])
                    yldm[2]= np.array([x2,y2])
                    yldm[3]= np.array([x3,y3])
                    
                    # print(type(yldm))
                    ypts= np.asarray([[yldm[0][0],yldm[0][1]],[yldm[1][0],yldm[1][1]],[yldm[3][0],yldm[3][1]],[yldm[2][0],yldm[2][1]]],dtype = np.float32)
                    print(ypts)
                    warped_ = four_point_transform(img, ypts)
                    cv2.imwrite(output_filename ,warped_)
                except:
                     print("loi")'''
'''img = cv2.imread("test/0005_00512_b.jpg")
    import json
    file_label_data = json.load(open('test/0005_00512_b.json','r'))
    print((file_label_data['shapes'][0]['points']))
    yolo_kps = np.array(file_label_data['shapes'][0]['points'])
    print(yolo_kps)
    tmp = copy.deepcopy(yolo_kps[3])
    yolo_kps[3]= yolo_kps[2]
    yolo_kps[2]= tmp
    # print(yolo_kps.shape)
    yldm = copy.deepcopy(yolo_kps)	
    print(yldm)
    ratio = 0.2
    x1,y1,x2,y2 = get_padding_align(yldm[1],yldm[2],ratio)
    x0,y0,x3,y3 = get_padding_align(yldm[0],yldm[3],ratio)
    yldm[0]= np.array([x0,y0])
    yldm[1]= np.array([x1,y1])
    yldm[2]= np.array([x2,y2])
    yldm[3]= np.array([x3,y3])
    
    # print(type(yldm))
    ypts= np.asarray([[yldm[0][0],yldm[0][1]],[yldm[1][0],yldm[1][1]],[yldm[3][0],yldm[3][1]],[yldm[2][0],yldm[2][1]]],dtype = np.float32)
    print(ypts)
    warped_ = four_point_transform(img, ypts)
    cv2.imwrite('output.jpg',warped_)'''