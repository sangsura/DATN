import argparse
from typing import Any
import copy
import easyocr
import cv2
import numpy as np
import onnxruntime as ort
# import torch

# from ultralytics.yolo.utils import ROOT, yaml_load
# from ultralytics.yolo.utils.checks import check_requirements, check_yaml
import yaml 
import re
import time
import torch
import cv2
from model.Mobilenet import *
from model.resnet import *
from collections import OrderedDict
import pickle
import numpy as np
from PIL import Image
from code_align import *
from ultralytics import YOLO

NUM_CLASSES = 1
NUM_KPS = 4

def static_resize(img, dh = 640, dw = 640):
    h,w,c = img.shape
    my_img = np.zeros((dh,dw,3),dtype=np.uint8)
    
    if h > w:
        nw = int(640*w/h)
        dim = (nw,640)
    else:
        nh = int(640*h/w)
        dim = (640,nh)
    rs_img = cv2.resize(img,dim, interpolation = cv2.INTER_AREA)
    rh, rw,_ = rs_img.shape
    # cv2.imwrite("aa.jpg",rs_img)
    my_img[0:rh,0:rw,:] = rs_img[:,:,:]
    # cv2.imwrite("a.jpg",my_img)
    return my_img
        

def yaml_load(file='data.yaml', append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        dict: YAML data and file name.
    """
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)

        # Add YAML filename to dict and return
        return {**yaml.safe_load(s), 'yaml_file': str(file)} if append_filename else yaml.safe_load(s)

kps_color=[(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,0)]

# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm
"""
Create patch from original input image by using bbox coordinate
"""

import cv2
import numpy as np

class Yolov8:

    def __init__(self, onnx_model, confidence_thres, iou_thres):
        """
        Initializes an instance of the Yolov8 class.

        Args:
            onnx_model: Path to the ONNX model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.onnx_model = onnx_model
        # self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # Load the class names from the COCO dataset
        # self.classes = yaml_load(check_yaml('ekyc_dataset.yaml'))['names']
        self.classes = yaml_load('face.yaml')['names']

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        self.session = ort.InferenceSession(self.onnx_model, providers=['CPUExecutionProvider']) #'CUDAExecutionProvider', 

        # Get the model inputs
        self.model_inputs = self.session.get_inputs()

        # Store the shape of the input for later use
        input_shape = self.model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

    def draw_detections(self, img, box, score, class_id,attribute,ocr):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f'{self.classes[class_id]}: {score:.2f}: {attribute}: {ocr}'

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                      cv2.FILLED)

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        '''for idx,kps in enumerate(keypoint):
            # print(kps[2])
            cv2.circle(img,(int(kps[0]),int(kps[1])),20,kps_color[idx],20)
            # cv2.putText(img, str(kps[2]), (int(kps[0]),int(kps[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, kps_color[idx], 1, cv2.LINE_AA)'''

    def preprocess(self, img):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Read the input image using OpenCV
        # self.img = cv2.imread(self.input_image)

        # Get the height and width of the input image
        # img_height, img_width = img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = static_resize(img)
        # cv2.imwrite("a.jpg",img)
        # img = cv2.resize(img, (img_width, img_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data

    def postprocess(self, input_image, output):
        # print(output[0].shape)
        
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        "output shape: 0, 1 ,2, 3: bbox, 4:4+num classes: conf score of bbox "
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))
        # print(outputs)
        # print(outputs.shape)
        # outputs = outputs[:,0:15]
        
        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []
        keypoints =[]

        # Calculate the scaling factors for the bounding box coordinates
        # x_factor = self.img_width / self.input_width
        # y_factor = self.img_height / self.input_height
        h,w,_ = input_image.shape
        if h > w:
            # nw = int(640*w/h)
            factor = h / self.input_height
        else:
            # nh = int(640*h/w)
            # dim = (640,nh)
            factor = w / self.input_width
        
        
        # Iterate over each row in the outputs array
        # print(rows)
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:4+NUM_CLASSES] # 11 class: 4:15
            kps_raw_datas = outputs[i][4+NUM_CLASSES:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)
            # print(max_score)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # print("here")
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * factor)#x_factor)
                top = int((y - h / 2) * factor)#y_factor)
                width = int(w * factor)#x_factor)
                height = int(h * factor)#y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
                kps = []
                for j in range (NUM_KPS): # num kps
                    x = kps_raw_datas[j*3]*factor#x_factor
                    y = kps_raw_datas[j*3+1]*factor#y_factor
                    kps_conf = kps_raw_datas[j*3+2]
                    kps.append([x,y,kps_conf])
                keypoints.append(kps)
        # Apply non-maximum suppression to filter out overlapping bounding boxes
        # print(boxes)
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        boxes_info = []
        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box_info = dict()
            box_info['box'] = boxes[i]
            box_info['score'] = scores[i]
            box_info['class_id'] = class_ids[i]
            box_info["keypoint"] = keypoints[i]
            boxes_info.append(box_info)

            # Draw the detection on the input image
            # self.draw_detections(input_image, box, score, class_id, keypoint)

        # Return the modified input image
        return boxes_info

    # def main(self):
    #     """
    #     Performs inference using an ONNX model and returns the output image with drawn detections.

    #     Returns:
    #         output_img: The output image with drawn detections.
    #     """
    #     # Create an inference session using the ONNX model and specify execution providers
        
    #     # Preprocess the image data
    #     img_data = self.preprocess()

    #     # Run inference using the preprocessed image data
    #     outputs = self.session.run(None, {self.model_inputs[0].name: img_data})

    #     # Perform post-processing on the outputs to obtain output image.
    #     output_img = self.postprocess(self.img, outputs)

    #     # Return the resulting output image
    #     return output_img
    def __call__(self, img):
        # Preprocess the image data
        img_data = self.preprocess(img)

        # Run inference using the preprocessed image data
        outputs = self.session.run(None, {self.model_inputs[0].name: img_data})

        # Perform post-processing on the outputs to obtain output image.
        boxes_info = self.postprocess(img, outputs)

        # Return the resulting output image
        output_img = img.copy()
        '''for box in boxes_info: 
        #     # print(box)
            self.draw_detections(output_img, box['box'], box['score'], box["class_id"], box['keypoint'])'''
        return boxes_info
class AttributeDetector_resnet():
    '''def __init__(self, weight='',size=(384,384)):
        # self.weight = weight
        # self.model = self.load_model(weight)
        self.model = self.load_mobilenetv3_model(weight,0)#'instance_att_classification_model/resnet18.glare'
        self.img_dh = size[0]
        self.img_dw = size[1]'''
    def __init__(self, weight='',size=(112,112)):
        # self.weight = weight
        # self.model = self.load_model(weight)
        self.model = self.load_resnet_model(weight,0)#'instance_att_classification_model/resnet18.glare'
        self.img_dh = size[0]
        self.img_dw = size[1]

    def load_resnet_model(self,weights_path, device = 1):
        model = ResNet18(num_classes=2)
        w = torch.load(weights_path)
        model.load_state_dict(w)
        
        model.eval()
        #print(model)
        return model.cuda(device)
   
    
    def resizePadding(self,img):
        desired_w = self.img_dw#width, height  # (width, height)
        desired_h = self.img_dh
        img_w, img_h = img.size  # old_size[0] is in (width, height) format
        ratio = 1.0 * img_w / img_h
        new_w = int(desired_h * ratio)
        new_w = new_w if desired_w == None else min(desired_w, new_w)
        img = img.resize((new_w, desired_h), Image.ANTIALIAS)

        # padding image
        if desired_w != None and desired_w > new_w:
            new_img = Image.new("RGB", (desired_w, desired_h), color=(255,255,255))
            new_img.paste(img, (0, 0))
            img = new_img
        img = np.array(img)/255.0
        img = img.transpose(2,0,1)
        # img = ToTensor()(img)
        # img = Normalize(mean, std)(img)
        return img
    def __call__(self,img):
        im_pil = Image.fromarray(img)
        data_tensor = self.resizePadding(im_pil)
        data_tensor = np.expand_dims(data_tensor,axis=0)
        data_tensor = torch.from_numpy(data_tensor).float().cuda(0)

        output = self.model(data_tensor)
        output= torch.softmax(output,1).cpu().tolist()
        return output[0]



'''import copy
import easyocr
reader = easyocr.Reader(['en'])
if __name__ == '__main__':
    video_path = '20130602135347-7.jpg'
    blur_path='blur_runs-48-6288.pt'
    occ_path='occ_runs-48-6288.pt'
    att_occ=AttributeDetector_resnet(occ_path)
    att_blur=AttributeDetector_resnet(blur_path)
    # Đường dẫn đến file video đầu ra
    output_video_path = 'out.jpg'
    img = cv2.imread(video_path)
    # Khởi tạo mô hình Yolov8
    model = Yolov8(onnx_model='yolov8l_lp.onnx', confidence_thres=0.5, iou_thres=0.5)

        # Thực hiện dự đoán
    boxes_info = model(img)

        # Vẽ bounding box và keypoint lên khung hình
    for box_info in boxes_info:
            box = box_info['box']
            score = box_info['score']
            class_id = box_info['class_id']
            key_point= box_info["keypoint"]
            print(key_point)
            yolo_kps = np.array(key_point)
            yolo_kps=yolo_kps[:,:2]
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
                   
            cv2.imwrite('a.jpg',warped_)
            check_occ = np.argmax(att_occ(warped_))
            check_blur=np.argmax(att_blur(warped_))
            print(check_occ)
            print(check_blur)
            check=''
            ocr=[]

            if check_occ==0 and check_blur==0:
                check = 'normal'
                result = reader.readtext(warped_)
                for detection in result:
                    ocr.append(detection[1])
            elif check_occ==0 and check_blur==1:
                check = 'blur'
            elif check_occ==1 and check_blur==0:
                check = 'occlusion'
            else:
                check = 'blur and occlusion'
            
            # Vẽ bounding box và keypoint
            model.draw_detections(img, box, score, class_id,check,ocr)
    # Ghi khung hình đã được vẽ vào video đầu ra
    cv2.imwrite(output_video_path,img)'''

'''# Đường dẫn đến file video đầu vào
    video_path = 'sample (online-video-cutter.com).mp4'

    # Đường dẫn đến file video đầu ra
    output_video_path = 'output_video13.mp4'

    # Khởi tạo mô hình Yolov8
    model = Yolov8(onnx_model='yolov8l_lp.onnx', confidence_thres=0.5, iou_thres=0.5)

    # Mở video đầu vào
    cap = cv2.VideoCapture(video_path)

    # Lấy thông số video (chiều rộng, chiều cao, tỷ lệ khung hình)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Tạo video writer để ghi video kết quả
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Đọc video từng khung hình và thực hiện dự đoán
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Thực hiện dự đoán
        boxes_info = model(frame)

        # Vẽ bounding box và keypoint lên khung hình
        for box_info in boxes_info:
            box = box_info['box']
            score = box_info['score']
            class_id = box_info['class_id']
            key_point= box_info["keypoint"]
            # Vẽ bounding box và keypoint
            model.draw_detections(frame, box, score, class_id,key_point)
    # Ghi khung hình đã được vẽ vào video đầu ra
        out.write(frame)
        print(frame)

    # Giải phóng tài nguyên
    cap.release()
    out.release()'''
        
    

    
  

