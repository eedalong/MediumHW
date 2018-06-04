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
predictor_path = 'models/landmarks_detector.dat';
detector = dlib.get_frontal_face_detector();
predictor = dlib.shape_predictor(predictor_path);
Mouth_Start = 48;
Mouth_End = 68;
cmap = [[0.0109,0.1379,0.96050],[0.0409,0.9379,0.46050],[0.0539,0.9379,0.9050],[0.8939,0.6379,0.8050]];
def LipMask2(image,shape):
    Mask_Points = [];
    InterpolateNum = 20;
    points_list1 = [48,49,50,51,52,53,54,64,63,62,61,60,48];
    points_list2 = [48,59,58,57,56,55,54,64,65,66,67,60,48];
    points_lists = [points_list1,points_list2];
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

    cv2.imwrite('/home/sensetime/dalong/mask.jpg',Mask_image);
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


def BeautifyLips(MouthImage,Choice):
    '''
    return the mouth image after processed with regard to color choice

    This will be implemented after Yi Liu implemented his matlab code

    and i will transform the code to python as soon as he finished that
    '''
    hsv_image = color.rgb2hsv(MouthImage);
    ratio = 0.15;
    hsv_image[:,:,0] = (1 - ratio) * hsv_image[:,:,0] + ratio * (cmap[Choice][0] - hsv_image[:,:,0]);
    hsv_image[:,:,1] = (1 - ratio) * hsv_image[:,:,1] + ratio * (cmap[Choice][1] - hsv_image[:,:,1]);
    hsv_image[:,:,2] = (1 - ratio) * hsv_image[:,:,2] + ratio * (cmap[Choice][2] - hsv_image[:,:,2]);

    Mouth = color.hsv2rgb(hsv_image);
    #print(Mouth);
    print('dalong log : after beautify');

    #cv2.imwrite('/home/sensetime/dalong/after_beautify.jpg',255*hsv_image[:,:,::-1])
    return np.array(Mouth * 255,dtype = np.uint8);

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
        image = BeautifyLips(image,choice);

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
    image_path = 'demos/1.jpg';

    image = io.imread(image_path);
    start = time.time();
    print('dalong log : into Beautify function');
    result = Beautify(image,3);
    cv2.imwrite('/home/sensetime/dalong/test.jpg',result[:,:,::-1]);
    print('dalong log : demo done it consumes {} seconds '.format(time.time() - start));


if __name__ == '__main__':
    #main();
    VideoDemo();





