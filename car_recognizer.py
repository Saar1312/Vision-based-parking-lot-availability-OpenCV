##################################################################################################################
#	University of Porto 
#	Faculty of Engineering
#	Computer Vision
#
# Project 1: Vision-based computation of parking lot occupancy
#
# Authors:
#	* Nuno Granja Fernandes up201107699
#	* Samuel Arleo Rodriguez up201600802
#	* Katja Hader up201602072
##################################################################################################################

import numpy as np
import cv2
from matplotlib import pyplot as plot
from sklearn.cluster import DBSCAN
import random as rd
import os
import sys

def numImages(list):
	tam = len(list)
	i = 0
	while list[i] is not None:
		i+=1
		if i == tam:
			break
	return i

# MatPlotLib method for printing images
def printImage(img1,img2=None,img3=None,img4=None):
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

# OpenCV method for printing images
def printImage(img):
	# Press ESC to exit
	if img is not None:
		cv2.imshow('res',img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	else: 
		print "Problem printing image."

def getChannels(img):
	return img[:,:,2],img[:,:,1],img[:,:,0]

def applyThreshold(gray,min_thr,max_thr):
	rows,col = gray.shape
	thr = np.zeros((rows,col)) # Creating a null matrix for copying matrix gray2
	np.copyto(thr,gray)
	for i in range(0,rows):
		for j in range(0,col):
			if min_thr <= thr[i,j] and thr[i,j] <= max_thr:
				thr[i,j] = 0
			else:
				thr[i,j] = 255	
	return thr

def thresholdImage(gray,thr_type,thr,block_size=None,img=None):

	""" Where thr_type in {1,2,3,4}
		1: Normal threshold
		2: Otsu
		3: Adaptive (mean)
		4: Adaptive (Gaussian)
		More thresholds: Using two thresholds taking into account that most pixels are from the floor 
			(Trying to don't erase black cars)
		5: Double threshold (using percentiles) 
		6: Double threshold (using manually set values)
	"""
	if thr_type == 1: 
		ret,thr = cv2.threshold(gray,thr,255,cv2.THRESH_BINARY)
		return thr
	elif thr_type == 2:
		ret,thr = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Black/red cars desapeared. Good for Segmentation of background
		return thr
	elif thr_type == 3:
		return cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,block_size,2) # Less noise, but can't recognize all cars
	elif thr_type == 4:
		return cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block_size,2) # More noise, more cars
	elif thr_type == 5:
		firstQ = np.percentile(gray2,65) # Return de value of the pixel that corresponds to the 65 percent of all sorted pixels in grayscale
		secondQ = np.percentile(gray2,50)
		thirdQ = np.percentile(gray2,35)
		return applyThreshold(gray,firstQ,thirdQ)
	elif thr_type == 6:
		return applyThreshold(gray,40,113)
	elif thr_type == 7:
		rows,col = img[:,:,0].shape
		r1,g1,b1 = getChannels(gray) # Actually is not grayscale but a BGR image (just a name)
		r2,g2,b2 = getChannels(img)
		res = np.zeros((rows,col))
		for i in range(0,rows):
			for j in range(0,col):
				rDif = abs(int(r1[i,j]) - int(r2[i,j]))
				gDif = abs(int(g1[i,j]) - int(g2[i,j]))
				bDif = abs(int(b1[i,j]) - int(b2[i,j]))
				if rDif >= thr or gDif >= thr or bDif >= thr:
					res[i,j] = 0
				else:
					res[i,j] = 255
		return res

	else:
		return None

def getEdges(gray,detector,min_thr=None,max_thr=None):
	"""
		Where detector in {1,2,3,4}
		1: Laplacian
		2: Sobelx
		3: Sobely
		4: Canny
		5: Sobelx with possitive and negative slope (in 2 negative slopes are lost) 
	"""
	if min_thr is None:
		min_thr = 100
		max_thr = 200
	if detector == 1:
		return cv2.Laplacian(gray,cv2.CV_64F)
	elif detector == 2:
		return cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=-1)
	elif detector == 3:
		return cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=-1)
	elif detector == 4:
		return cv2.Canny(gray,min_thr,max_thr)  # Canny(min_thresh,max_thresh) (threshold not to the intensity but to the
												# intensity gradient -value that measures how different is a pixel to its neighbors-)
	elif detector == 5:
		sobelx64f = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
		abs_sobel64f = np.absolute(sobelx64f)
		return np.uint8(abs_sobel64f)

def dbscan(points,eps,min_samples):
	db = DBSCAN(eps=eps, min_samples=min_samples).fit(points) # eps=5 min_samples = 80

	# Labeling pixels by cluster
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_

	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

	# Creating list of clusters
	return [points[labels == i] for i in xrange(n_clusters_)]

def getCentroids(clusters,img):
	n = 0
	# centroid will store the coordinates of the center of each cluster 
	centroids = np.zeros((len(clusters),2),dtype=np.int_)
	for c in clusters:
		x,y = (int(sum(c[:,0])/len(c[:,0])),int(sum(c[:,1])/len(c[:,1])))
		r,g,b = rd.randint(0,255),rd.randint(0,255),rd.randint(0,255)
		centroids[n,0],centroids[n,1] = x,y
		n = n + 1
		cv2.circle(img,(y,x),7,(r,g,b),-1)
	return centroids,img

def paintClusters(img,clusters):
	# Painting clusters
	for c in clusters:
		r,g,b = rd.randint(0,255),rd.randint(0,255),rd.randint(0,255)
		for pixel in c:
			img[int(pixel[0]),int(pixel[1]),:] = b,g,r
	return img
			

# Empty and non-empty parking lot image
img1 = cv2.imread("img8.jpg")
img2 = cv2.imread("img9.jpg")
if img1 is None or img2 is None:
	print("It was not possible to load the images. Please check paths.")
	sys.exit()

# Blurring empty parking lot image
blur = cv2.GaussianBlur(img1,(7,7),0)

# Dimensions of the image
rows,col = img1[:,:,0].shape

# Images for storing results
res2 = img2.copy()
res3 = img2.copy()
res4 = img2.copy()
res5 = img2.copy()

# Auxiliar image for storing results
res = np.zeros((rows,col))

# Gray scale image of the non-empty parking lot
gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# Edges on non-empty parking lot using Canny edge detector 
edges = getEdges(gray,4,50,200)

# Threshold for every channel
threshold = 40

# Applying color threshold: It takes each channel of the empty parking lot image 
# and compares each pixel with the same channel on the non-empty park. If this difference
# is more than the threshold, then the pixel is considered as a car pixel and the pixel
# on the same position in the binary image "res" is painted black. This allows to recognize
# cars of almost any color, however cars with intisities similar to the background are
# hard to paint.
res = thresholdImage(img2,7,threshold,img=blur)

# Adding edges to the thresholded image to separate cars
res = res+edges

# Getting indexes of car pixels (black pixels) on res
cars_pixels = np.where(res == 0)

# Number of car pixels
num_pix = cars_pixels[0].size

# Declaring matrix with the same number of rows as black pixels the image has
X = np.zeros((num_pix,2)) 

# Putting coordinates of pixels in X
for i in range(0,num_pix):
	X[i,:] = [cars_pixels[0][i],cars_pixels[1][i]]

# Applying dbscan clustering to X (pixels positions) where:
# eps: distance from current centroid to others
# min_samples: min number of pixels that must be at a distance of less or equal than eps
# to consider the current pixel as a centroid
eps = 5
min_samples = 80

clusters = dbscan(X,eps,min_samples)

# Centroids stores the coordinates of the center of each cluster 
centroids,res2 = getCentroids(clusters,res2)

if centroids.size == 0:
	print("No clusters were created. Please change dbscan parameters.")
	sys.exit()

# Trying to cluster the centroids
eps2 = 30
min_samples2 = 1

# Creating list of clusters
clusters2 = dbscan(centroids,eps2,min_samples2)
#clusters2 = (clusters2[:,1],clusters2[:,0])

# centroid will store the coordinates of the center of each cluster 
centroids2,res3 = getCentroids(clusters2,res3)

# Trying to cluster the centroids again
eps3 = 45
min_samples3 = 1

# Creating list of clusters
clusters3 = dbscan(centroids2,eps3,min_samples3)
#clusters2 = (clusters2[:,1],clusters2[:,0])

# centroid will store the coordinates of the center of each cluster 
centroids3,res5 = getCentroids(clusters3,res5)

# Painting clusters
res4 = paintClusters(res4,clusters)

printImage(res)
printImage(res2)
printImage(res3)
printImage(res5)
printImage(res4)

# Saving resulting image
cv2.imwrite("PRUEBAres_thr_"+str(threshold)+"_eps1_"+str(eps)+"_min1_"+str(min_samples)+
						"_eps2_"+str(eps2)+"_min2_"+str(min_samples2)+".jpg",res3)


# If you want to calculate an approximate number of empty spaces write the number of 
# available spaces on the empty parking lot picture in the following variable:
"""
capacity = 76
num_cars = len(clusters3)
if num_cars >= capacity :
	print("There is no place where to park.")
else:
	print("There are available spaces.")
"""

print(len(centroids2))
print(len(centroids3))