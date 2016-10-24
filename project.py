import numpy as np
import cv2
from matplotlib import pyplot as plot

def swapRB(img):
	return np.dstack([img[:,:,2],img[:,:,1],img[:,:,0]])

def numImages(list):
	tam = len(list)
	i = 0
	while list[i] is not None:
		i+=1
		if i == tam:
			break
	return i

def printImages(img1,img2=None,img3=None,img4=None):
	images = [img1,img2,img3,img4]
	n = numImages(images)
	for i in range(0,n):
		if n<3:
			plot.subplot(1,n,i+1),plot.imshow(images[i],cmap='Greys_r')
			plot.title(str(i)), plot.xticks([]), plot.yticks([])
		else:
			plot.subplot(2,2,i+1),plot.imshow(images[i],cmap='Greys_r')
			plot.title(str(i)), plot.xticks([]), plot.yticks([])
	plot.show()

# OpenCV method
def printImages2(img):
	# Press ESC to exit
	cv2.imshow('res',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



img = cv2.imread("img3.jpg")

#b,g,r = cv2.split(img)
#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


zeros = np.zeros((img[:,:,0].shape[0],img[:,:,0].shape[1])) # Matrix made of 0's with the dimensions of one channel of img
r = np.dstack([img[:,:,0],zeros,zeros])
g = np.dstack([zeros,img[:,:,1],zeros])
b = np.dstack([zeros,zeros,img[:,:,2]])

# Attributes numpy array: ndim, size, shape, dtype, itemsize

#Merging channels conversely to make BGR->RGB
#image = swapRB(img)

#Saving images of each channel
#cv2.imwrite("r.jpg",r)
#cv2.imwrite("g.jpg",g)
#cv2.imwrite("b.jpg",b)


image = swapRB(img)

# Gaussian filter     cv2.GaussianBlur(image,(kernel_size_x,kernel_size_y),x_std_dev,y_std_dev) if 0 both std dev are obtained by kernel size
blur = cv2.GaussianBlur(image,(3,3),0)

gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


# WITH BLURRING (gaussian smoothing)
# Normal threshold still allows to see a difference between a black car and an empty space
ret,mask = cv2.threshold(gray,70,255,cv2.THRESH_BINARY)

# Otsu threshold
ret2,mask2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Black/red cars desapeared. Good for Segmentation of background

# Adaptatives threasholds
# Mean threshold
thr1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,23,2) # Menos ruido, no reconoce algunos carros
thr2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,23,2) # Mas ruido, peor reconoce mas

# WITHOUT BLURRING
# Normal threshold still allows to see a difference between a black car and an empty space
ret3,mask3 = cv2.threshold(gray2,70,255,cv2.THRESH_BINARY)

# Otsu threshold
ret4,mask4 = cv2.threshold(gray2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Black/red cars desapeared. Good for Segmentation of background

# Adaptatives threasholds
# Mean threshold

thr3 = cv2.adaptiveThreshold(gray2,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,23,2) # Menos ruido, no reconoce algunos carros
thr4 = cv2.adaptiveThreshold(gray2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,23,2) # Mas ruido, peor reconoce mas

printImages(thr1,thr3)





