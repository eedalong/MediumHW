import sys
import os
import dlib
from skimage import io
from skimage import transform
import LipMask
import time
import cv2;
import numpy as np
import skimage.color as color
from skimage import transform  as sktransform
from PIL import ImageEnhance
from PIL import Image
predictor_path = 'models/landmarks_detector.dat';
detector = dlib.get_frontal_face_detector();
predictor = dlib.shape_predictor(predictor_path);
Mouth_Start = 48;
Mouth_End = 68;
cmap = [[0.0109,0.1379,0.96050],[0.0409,0.9379,0.46050],[0.0539,0.9379,0.9050],[0.8939,0.6379,0.8050]];

YSL_RGB = [[204,139,147],[190,132,110],[221,165,190],[216,155,171],[230,106,158],[199,82,107],[181,118,135],[216,122,122],[193,131,154],[255,124,182]];

points_list1 = [48,49,50,51,52,53,54,64,63,62,61,60,48];
points_list2 = [48,59,58,57,56,55,54,64,65,66,67,60,48];
points_lists = [points_list1,points_list2];

def LipMask2(image,shape):
    Mask_Points = [];
    InterpolateNum = 20;
    global points_lists,points_list1,points_list2;

    Mask_image = np.zeros(image.shape,dtype = np.uint8);
    for points_list in points_lists:
        roi_corners = [[],];
        for index in range(0,len(points_list) - 3):
            for t in range(InterpolateNum):
                t = t * 1.0/ InterpolateNum;
                x = (1-t )**2 * shape.part(points_list[index]).x + 2*t*(1-t)*shape.part(points_list[index+1]).x + t**2 * shape.part(points_list[index+2]).x;
                y = (1-t)**2 * shape.part(points_list[index]).y + 2*t*(1-t)*shape.part(points_list[index+1]).y + t**2 * shape.part(points_list[index+2]).y;
                Mask_Points.append((int(x),int(y)));
                roi_corners[0].append((int(x),int(y)));
        roi_corners = np.array(roi_corners,dtype = np.uint32);
        cv2.fillPoly(Mask_image,np.array(roi_corners,dtype = np.int32),(255,255,255));

    cv2.imwrite('/home/yuanxl/lipmask/mask.jpg',Mask_image);
    tmp_image = cv2.bitwise_and(Mask_image,image);
    rest_image = image - tmp_image;
    return Mask_Points,image,Mask_image,rest_image;

def draw_points2(image,Mask_Points):
    for index in range(len(Mask_Points)):
        cv2.circle(image,(Mask_Points[index][0],Mask_Points[index][1]),radius = 1,color =(0,255,0),thickness = -1);
    return image;

def draw_points(image,shape):
    for index in range(Mouth_Start,Mouth_End):
        cv2.circle(image,(shape.part(index).x,shape.part(index).y),radius = 1,color =(0,255,0),thickness = -1);
    return image;


def GetMouseLocation(image,shape):
    '''
    return MouseLoacation (left,top,right,bottom)
    This is not implemented for now because the i havent download the model
    and The Download is undergoing now
    '''
    left = shape.part(Mouth_Start).x;
    right = shape.part(Mouth_Start).x;
    top = shape.part(Mouth_Start).y;
    bottom = shape.part(Mouth_Start).y;
    for index in range(Mouth_Start,Mouth_End):
        if shape.part(index).x < left:
            left = shape.part(index).x;
        if shape.part(index).x > right:
            right = shape.part(index).x;
        if shape.part(index).y < top:
            top = shape.part(index).y;
        if shape.part(index).y > bottom:
            bottom = shape.part(index).y
    #expand the bbox for 1.5
    '''
    left = max(0,left - 0.25 *(right - left));
    right = min(image.shape[1],right + 0.25 *(right - left));
    top = max(0,top - 0.25* (bottom - top));
    bottom =  min(image.shape[0],bottom + 0.25 * (bottom - top));
    '''
    # return the bbox
    return [left  ,top,right,bottom];
def EnhanceMouth(MouthImage):
    image = Image.fromarray(MouthImage);
    enh_col = ImageEnhance.Color(image);
    color = 1.2;
    image_colored = enh_col.enhance(color);
    return np.asarray(image_colored);

def BeautifyLips(MouthImage,Choice):
    '''
    return the mouth image after processed with regard to color choice

    This will be implemented after Yi Liu implemented his matlab code

    and i will transform the code to python as soon as he finished that
    '''
    hsv_image = color.rgb2hsv(MouthImage);
    ratio = 0.25;
    hsv_image[:,:,0] = (1 - ratio) * hsv_image[:,:,0] + ratio * (cmap[Choice][0] - hsv_image[:,:,0]);
    hsv_image[:,:,1] = (1 - ratio) * hsv_image[:,:,1] + ratio * (cmap[Choice][1] - hsv_image[:,:,1]);
    hsv_image[:,:,2] = (1 - ratio) * hsv_image[:,:,2] + ratio * (cmap[Choice][2] - hsv_image[:,:,2]);

    Mouth = color.hsv2rgb(hsv_image);
    #print(Mouth);
    print('dalong log : after beautify');

    #cv2.imwrite('/home/yuanxl/after_beautify.jpg',255*hsv_image[:,:,::-1])
    return np.array(Mouth * 255,dtype = np.uint8);
def GetRect(shape):
    global points_lists;
    x1 = y1 = 100000;
    x2 = y2 = 0;
    for points_list in points_lists:
        for point in points_list:
            if shape.part(point).x < x1:
                x1 = shape.part(point).x;
            if shape.part(point).x > x2:
                x2 = shape.part(point).x;
            if shape.part(point).y < y1:
                y1 = shape.part(point).y;
            if shape.part(point).y > y2:
                y2 = shape.part(point).y;

    return [x1,y1,x2,y2];
def GetGaussMap(rect):
    sigma = min((rect[2] - rect[0]) / 1.5 ,(rect[3] - rect[1]) / 1.5);
    gauss_map = np.zeros((rect[3] - rect[1],rect[2] - rect[0]));
    center_x = (rect[0] + rect[2] )/ 2;
    center_y = (rect[1] + rect[3]) / 2;
    for i in range(rect[3] -rect[1]):
        for j in range(rect[2] - rect[0]):
            dist = np.sqrt((i + rect[1] -center_y)**2 + (j + rect[0] - center_x)**2);
            gauss_map[i][j] = np.exp(-0.5 * dist**2 / sigma**2);
    return gauss_map;

def AddGaussian(MaskImage,shape):

    rect = GetRect(shape);
    Gauss_map = GetGaussMap(rect);
    #print('dalong log : check gauss map = {}'.format(Gauss_map));
    Gauss_map = np.expand_dims(Gauss_map,axis = 2);
    Gauss_map = np.concatenate((Gauss_map,Gauss_map,Gauss_map),axis = 2);
    crop_image = MaskImage[rect[1]:rect[3],rect[0]:rect[2],:];
    crop_image = Gauss_map * crop_image;
    MaskImage[rect[1]:rect[3],rect[0]:rect[2],:] = crop_image;
    return MaskImage;

def BeautifyLips2(MouthImage,Choice,shape):
    alphaA = 1;

    MouthImage = MouthImage / 255.0;
    MaskImage = np.zeros(MouthImage.shape);
    alphaB = np.zeros(MouthImage.shape);
    alphaB[:,:,:] = 0.3;

    MaskImage[:,:,0] = YSL_RGB[Choice][0] / 255.0;
    MaskImage[:,:,1] = YSL_RGB[Choice][1] / 255.0;
    MaskImage[:,:,2] = YSL_RGB[Choice][2] / 255.0;
    alphaB = AddGaussian(alphaB,shape);
    MouthImage = (alphaA * MouthImage *(1.0 - alphaB) + MaskImage * alphaB)  / (alphaA + alphaB -  alphaA * alphaB);
    MouthImage = np.array(MouthImage * 255,dtype = np.uint8);
    MouthImage = EnhanceMouth(MouthImage);
    return MouthImage;
def Beautify(image,choice):
    dets = detector(image,1);
    print('dalong log : Number of faces detected  = {}'.format(len(dets)));
    for k,d in enumerate(dets):
        print('dalong log : Detection {}: Left: {} Top: {} Right: {} Bottom: {}'.format(k,d.left(),d.top(),d.right(),d.bottom()));
        shape = predictor(image,d);
        '''
        MouthLocation = GetMouseLocation(image,shape);
        MouthImage = image[MouthLocation[1]:MouthLocation[3],MouthLocation[0]:MouthLocation[2],:].copy();
        #cv2.imwrite('/home/sensetime/dalong/mouth.jpg',MouthImage[:,:,::-1])
        MouseMask = LipMask.Lip_Segmentation(MouthImage);

        #cv2.imwrite('/home/sensetime/dalong/mouth_mask.jpg',(MouseMask*255));
        beautiful_mouth = BeautifyLips(MouthImage.copy(),choice);

        #cv2.imwrite('/home/sensetime/dalong/mouth2.jpg',beautiful_mouth[:,:,::-1])
        for row in range(MouthImage.shape[0]):
            for col in range(MouthImage.shape[1]):
                if MouseMask[row][col] < 0.5:
                    MouthImage[row,col,:] = beautiful_mouth[row,col,:];


        #cv2.imwrite('/home/sensetime/dalong/mouth3.jpg',MouthImage[:,:,::-1])
        image[MouthLocation[1]:MouthLocation[3],MouthLocation[0]:MouthLocation[2],:] = MouthImage;
        #image = draw_points(image,shape);
        '''

        #tmp_image = BeautifyLips(image,choice);
        mask,image,Mask_image,rest_image = LipMask2(image.copy(),shape);
        image = BeautifyLips2(image,choice,shape);

        image = cv2.bitwise_and(Mask_image,image);
        image = image + rest_image;
        #image = draw_points2(image,mask);
    return image ;

def VideoDemo():
    cap = cv2.VideoCapture(0);
    ret = True;
    frame_index = 0;
    while(ret):
        ret,frame = cap.read();
        if frame_index < 3:
            frame_index = frame_index +1 ;

        frame_index = 0;
        result = Beautify(frame[:,:,::-1].copy(),3);
        cv2.imshow('test_win',result[:,:,::-1]);
        cv2.waitKey(10);

        cv2.imwrite('/home/sensetime/dalong/test.jpg',result[:,:,::-1]);
def main():
    image_path = 'demos/2.jpg';

    image = io.imread(image_path);
    start = time.time();
    print('dalong log : into Beautify function');
    for index in range(len(YSL_RGB)):
        result = Beautify(image,index);
        cv2.imwrite('/home/yuanxl/MediumHW/results/test'+str(index)+'.jpg',result[:,:,::-1]);
        print('dalong log : demo done it consumes {} seconds '.format(time.time() - start));


if __name__ == '__main__':
    main();
    #VideoDemo();






