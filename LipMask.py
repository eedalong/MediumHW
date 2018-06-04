import skimage
import sklearn
import numpy as np
import skfuzzy as fuzz
from skimage import io
import time
from skimage import transform
from skimage import exposure
import cv2
FLIP = 0;
NONE_FLIP = 1;
def PseduoHUE(image):
	'''
	input image is an numpy array with shape WxHx3
	with channel order RGB
	'''
	ans = image[:,:,0] / (image[:,:,0] + image[:,:,1] + 1e-5);
	return ans

def prepare_inputs(image):
	'''
	prepare_inputs for FCM
	'''
	if len(image.shape) ==2:
		return image.reshape((1,image.shape[0]*image.shape[1]));
	else:
		print('dalong log : sorry ,the other type is not supported now ');

def get_prob(u):
	if u.shape[0] == 2:
		return u;
	else :
		print('dalong log : sorry, the other type is nor supported now ');


def display(image_prob):
	tmp = image_prob[0,:] - image_prob[1,:];
	c = tmp.copy();
	c[tmp < 0] = FLIP;
	c[tmp > 0] = NONE_FLIP ;
	if np.sum(c) < 0.5 * tmp.shape[0]:
		c[tmp < 0] = NONE_FLIP;
		c[tmp > 0] = FLIP ;
	return c;

def FCM(image):
	'''
    inputs is an numpy array with shape WxHxN
	'''
	inputs = prepare_inputs(image);
	cntr,u,_,_,_,_,_ = fuzz.cluster.cmeans(inputs,2,2,error = 0.0005,maxiter = 2000);
	prob = get_prob(np.array(u));
	prob = display(prob)
	image_prob = prob.reshape((image.shape[0],image.shape[1]));
	return image_prob;

def get_xm(image_prob):
	xm = 0;
	for index in range(image_prob.shape[1]):
		xm = xm + index * np.sum(image_prob[:,index]);
	xm  = xm / np.sum(image_prob);
	return xm ;
def get_ym(image_prob):
	ym = 0;
	for index in range(image_prob.shape[0]):
		ym = ym + index * np.sum(image_prob[index,:]);
	ym  =ym / np.sum(image_prob);
	return ym ;

def get_theta(center,image_prob):
	u11 = 0;
	u20 = 0;
	u02 = 0;
	xm,ym = center[0],center[1];
	for row in range(image_prob.shape[0]):
		for col in range(image_prob.shape[1]):
			u11 = u11 + (col - xm )*(row - ym) * image_prob[row][col];
			u20 = u20 + (col - xm)**2 * image_prob[row][col];
			u02 = u02 + (row - ym)**2 * image_prob[row][col];
	theta = np.arctan(2*u11 / (u20 - u02));
	return theta;
def get_minoraxis(center,theta,image_prob):
	xm.ym = center[0],center[1];
	Ix = Iy = 0;
	for row in range(image.shape[0]):
		for col in range(image_prob.shape[1]):
			Ix = Ix + ((row - ym ) * np.cos(theta) - (col - xm) * np.sin(theta))**2 * image_prob[row][col];
			Iy = Iy + ((row - ym) * np.sin(theta) + (col - xm) * np.cos(theta))**2 * image_prob[row][col];
	xa = (4 / np.pi) ** 0.25 *(Iy**3 / Ix) **0.125;
	ya = (4 / np.pi) **0.25 * (Ix**3 / Iy) **0.125;
	return (xa,ya);

def get_dist(pointA,pointB):
	return np.sqrt(np.sum((pointA - pointB)**2));

def get_mask(axis,center,size):

	mask = np.zeros(size);
	for row  in range(size[0]):
		for col in range(size[1]):
			if get_dist(np.array([row,col]),center) <= 2 * axis[0]:
				mask[row][col] = 1;
	return mask;
def Post_Ellipse(image,image_prob):
	xm = get_xm(image_prob);
	ym = get_ym(image_prob);
	theta = get_theta(np.array([xm,ym]),image_prob);
	axis = get_minoraxis(np.array([xm,ym]), theta,image_prob);
	mask = get_mask(axis,np.array([pointA,pointB]),[image_prob.shape[0],image_prob.shape[1]]);
	image = image * mask;
	return image;


def Lip_Segmentation(image):
        save_image = image.copy();
        image = transform.resize(image,(240,320));
	image = PseduoHUE(image);
	#cv2.imwrite('/home/sensetime/dalong/after_hue.bmp',(image * 255).astype('uint8'));
        image_prob = FCM(image);
        image_prob = transform.resize(image_prob,(save_image.shape[0],save_image.shape[1]));
	#cv2.imwrite('/home/sensetime/dalong/after_fcm.bmp',(image_prob.astype('uint8') * 255).astype('uint8'));

        return image_prob.astype('uint8');


def GaussianMask(image):
	pass
def main() :
        FLIP = 0;
        NONE_FLIP = 1;
        image_path = 'demos/5.jpg';
	image = io.imread(image_path);
	image = transform.resize(image,(image.shape[0],image.shape[1]));
	ans = Lip_Segmentation(image);
	cv2.imwrite('result.jpg',(255*ans).astype('uint8'));
	return ;
if __name__  == '__main__':
	main();

