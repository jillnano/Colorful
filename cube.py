#!usr/bin/python
#coding=utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os
import cv2
import random
import string
import numpy as np
from glob import glob
from optparse import OptionParser
from PIL import Image, ImageDraw

import closed_form_matting

'''
def findWay(img, x, y, color, size = 5):
	p1 = img[max(x - 5, 0)][y]
	p2 = img[min(x + 5, img.shape[0] - 1)][y]
	p3 = img[x][max(y - 5, 0)]
	p4 = img[x][min(y + 5, img.shape[1] - 1)]
	if p1 >= 128 and p2 >= 128 and p3 >= 128 and p4 >= 128:
		img[x][y] = 128
		return False
	return True

def closeArea(originname, filename):
	step = 10
	areaData = []

	img = cv2.imread(originname)
	origin_gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	img = cv2.imread(filename)
	totalNum = img.shape[0] * img.shape[1]
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# threshold_img = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY)[1]
	mask = np.zeros((gray_img.shape[0]+2, gray_img.shape[1]+2), np.uint8)
	# seed_pt = (1, 1)
	consider_color = 50
	decide_color = 100
	for x in xrange(gray_img.shape[0]):
		for y in xrange(gray_img.shape[1]):
			if gray_img[x][y] == 0:
				# tmp_img = gray_img.copy()
				seed_pt = (y, x)
				# print color, seed_pt
				mask[:] = 0
				cv2.floodFill(gray_img, mask, seed_pt, consider_color, 0, 0, 8)
				pointList = np.where(gray_img == consider_color)
				cs = set()
				for posX, posY in zip(list(pointList[0]), list(pointList[1])):
					cs.add(int(origin_gray_img[posX][posY] / step) * step)
				num = np.sum(gray_img == consider_color)
				mask[:] = 0
				rate = num / float(totalNum)
				# print len(cs), num, rate
				if rate < 0.01:
					cv2.floodFill(gray_img, mask, seed_pt, 128, 0, 0, 8)
				elif len(cs) >= 3 and rate < 0.1:
					cv2.floodFill(gray_img, mask, seed_pt, 255, 0, 0, 8)
				else:
					cv2.floodFill(gray_img, mask, seed_pt, decide_color, 0, 0, 8)
					# print np.sum(gray_img == 0), num
	gray_img[np.where(gray_img == decide_color)] = 0
	cv2.imwrite('ret4.png', gray_img)

def colorful(filename, alpha):
	origin_img = cv2.imread(filename)
	gray_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
	bg_img = cv2.imread('bg2.jpg')
	new_img = np.zeros((origin_img.shape[0], origin_img.shape[1], origin_img.shape[2]), np.uint8)
	new_img[:] = 255
	alpha_img = cv2.cvtColor(cv2.imread(alpha), cv2.COLOR_BGR2GRAY)
	# threshold_img = cv2.threshold(alpha_img, 100, 255, cv2.THRESH_BINARY)[1]
	for x in xrange(alpha_img.shape[0]):
		for y in xrange(alpha_img.shape[1]):
			ti = alpha_img[x][y]
			if ti > 250:
				# rate = ti / 255.0
				new_img[x][y] = [ti, ti, ti]
			else:
				new_img[x][y] = bg_img[x][y]
	cv2.imwrite('ret5.png', new_img)
	# cv2.imwrite('ret6.png', gray_img)
	# cv2.imwrite('ret7.png', new_img)
'''

def process_image(imagename, resultname = None, edgeThreshold = 100):
	""" Process an image and save the results in a file. """
	img = cv2.imread(imagename)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	sift = cv2.xfeatures2d.SIFT_create(edgeThreshold = edgeThreshold)
	kpsList = []
	kps , des = sift.detectAndCompute(gray, None)
	for kp in kps:
		kpsList.append([kp.pt[0], kp.pt[1], kp.response, kp.angle])
	kpsList = np.array(kpsList)
	ret = np.hstack((kpsList, des))
	if resultname:
		np.savetxt(resultname, ret)
	return ret

def read_features(filename):
	""" Read feature properties and return in matrix form. """
	if isinstance(filename, (str, unicode)):
		f = np.loadtxt(filename)
	else:
		f = filename
	return f[:,:4],f[:,4:] # feature locations, descriptors

def mixClone(objImg, bgImg, objRate = 1.0):
	# Read images : src image will be cloned into dst
	if objRate > 0:
		objImg = objImg * objRate
		objImg = objImg.astype(np.uint8)
	else:
		bgImg = bgImg * abs(objRate)
		bgImg = bgImg.astype(np.uint8)
	objHeight, objWidth = objImg.shape[0:2]
	bgHeight, bgWidth = bgImg.shape[0:2]
	bgRate = max(objWidth / float(bgWidth), objHeight / float(bgHeight))
	# print obj.shape

	# Create an all white mask
	mask = 255 * np.ones(objImg.shape, objImg.dtype)

	bgImg = cv2.resize(bgImg, (int(bgWidth * bgRate) + 1, int(bgHeight * bgRate) + 1))
	bgHeight, bgWidth = bgImg.shape[0:2]
	# cv2.imwrite('newbg.jpg', bgImg)

	# The location of the center of the src in the dst
	height, width, _ = bgImg.shape
	center = (width / 2, height / 2)
	print 'obj:', objImg.shape
	print 'bg:', bgImg.shape
	print 'center:', center
	# Seamlessly clone src into dst and put the results in output
	# normal_clone = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
	mixed_clone = cv2.seamlessClone(objImg, bgImg, mask, center, cv2.MIXED_CLONE)
	# monochrome_transfer = cv2.seamlessClone(obj, im, mask, center, cv2.MONOCHROME_TRANSFER)
	mixed_clone = mixed_clone[center[1] - (objHeight / 2): center[1] + (objHeight / 2), center[0] - (objWidth / 2): center[0] + (objWidth / 2)]
	print 'out:', mixed_clone.shape
	return mixed_clone

	# Write results
	# mixed_clone = blue2red(mixed_clone)
	# mixed_clone = red2blue(mixed_clone, 156, 180)
	# mixed_clone = red2blue(mixed_clone, 0, 10)
	# mixed_clone = red2blue(mixed_clone, 0, 10)
	# mixed_clone = histYUV(mixed_clone)
	# mixed_clone = histColor(mixed_clone)
	# cv2.imwrite(output, mixed_clone)
	# cv2.namedWindow(output, cv2.WINDOW_NORMAL)
	# cv2.imshow(output, mixed_clone)
	# cv2.waitKey(0)

def matting_with_trimap(source, trimap, taskId):
	image = cv2.imread(source, cv2.IMREAD_COLOR) / 255.0
	trimap = cv2.imread(trimap, cv2.IMREAD_GRAYSCALE) / 255.0

	alpha = closed_form_matting.closed_form_matting_with_trimap(image, trimap)
	alphaname = 'tmp/' + taskId + ''.join([random.choice(string.lowercase) for _ in xrange(5)]) + '.jpg'
	cv2.imwrite(alphaname, alpha * 255.0)
	return alphaname

def makeSift(imname, taskId):
	# imname = 'wz.jpeg'
	img_origin = Image.open(imname)
	img = img_origin.convert('L')
	point_size = max(img.size) / 150
	# im1 = np.array(img)
	sift = process_image(imname)
	l1, d1 = read_features(sift)
	# print l1.shape, point_size
	newImg = Image.new('L', img.size, color = 0)
	imgDraw = ImageDraw.Draw(newImg)
	tx = l1[:,0]
	ty = l1[:,1]
	for i in xrange(len(tx)):
		imgDraw.ellipse((tx[i] - point_size, ty[i] - point_size, tx[i] + point_size, ty[i] + point_size), fill = 255)
	# newImg.save('ret.png')
	# newImgData = newImg.load()
	# fillPos = []
	# for x in xrange(img.size[0] - point_size):
	# 	for y in xrange(img.size[1] - point_size):
	# 		if x > point_size and y > point_size:
	# 			p1 = newImgData[x - point_size, y]
	# 			p2 = newImgData[x + point_size, y]
	# 			p3 = newImgData[x, y - point_size]
	# 			p4 = newImgData[x, y + point_size]
	# 			if newImgData[x, y] == 0:
	# 				if p1 == p2 == p3 == p4 == 255:
	# 					fillPos.append([x, y])
	# 					# pass
	# 				elif (p1 == p2 == 255) or (p3 == p4 == 255):
	# 					fillPos.append([x, y])
	# 					# pass

	# for x, y in fillPos:
	# 	imgDraw.ellipse((x - point_size / 2, y - point_size / 2, x + point_size / 2, y + point_size / 2), fill = 255)
	# newImg.save('ret2.png')

	# print len(fillPos)
	cv2_img = np.array(newImg)
	cv2_img = makeEage(np.array(img_origin), cv2_img)
	inner_map = cv2.erode(cv2_img, np.ones((point_size, point_size), np.uint8), iterations = 3)

	# cv2_img = cv2.imread('ret2.png')
	# return inner_map + (cv2_img - inner_map) / 2
	trimapname = 'tmp/' + taskId + ''.join([random.choice(string.lowercase) for _ in xrange(5)]) + '.jpg'
	trimap = inner_map + (cv2_img - inner_map) / 2
	# trimap = makeEage(np.array(img_origin), trimap)
	cv2.imwrite(trimapname, trimap)
	return trimapname

def red2blue(img, hmin = 156, hmax = 180):
	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	lower_blue = np.array([hmin,43,46])
	upper_blue = np.array([hmax,255,255])
	# print img_hsv.shape

	mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
	cv2.imwrite('mask.jpg', mask)

	height, width = img.shape[0:2]

	for y in xrange(height):
		for x in xrange(width):
			if mask[y][x] != 0:
				b, g, r = img[y][x]
				img[y][x] = np.array([r, g, b], np.uint8)
	return img

def blue2red(img):
	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	lower_blue = np.array([78,43,46])
	upper_blue = np.array([155,255,255])
	# print img_hsv.shape

	mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
	# cv2.imwrite('mask.jpg', mask)

	height, width = img.shape[0:2]

	for y in xrange(height):
		for x in xrange(width):
			if mask[y][x] != 0:
				b, g, r = img[y][x]
				img[y][x] = np.array([r, g, b], np.uint8)
	return img

def histYUV(img):
	imgYUV = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

	channelsYUV = cv2.split(imgYUV)
	channelsYUV[0] = cv2.equalizeHist(channelsYUV[0])

	channels = cv2.merge(channelsYUV)
	return cv2.cvtColor(channels, cv2.COLOR_YCrCb2BGR)

def makeEage(img, trimap):
	eage = cv2.Canny(img, 200, 255)
	h, w = eage.shape[0:2]
	for y in xrange(h):
		for x in xrange(w):
			if eage[y][x] > 200:
				trimap[y][x] = eage[y][x]
	return trimap

def histColor(img):
	(b, g, r) = cv2.split(img)
	bH = cv2.equalizeHist(b)
	gH = cv2.equalizeHist(g)
	rH = cv2.equalizeHist(r)
	# 合并每一个通道
	return cv2.merge((bH, gH, rH))

def mixSift(img, img_gray, img_origin):
	h, w = img.shape[0:2]
	for y in xrange(h):
		for x in xrange(w):
			if img_gray[y][x] > 0:
				rate = img_gray[y][x] / 255.0
				img[y][x] = img[y][x] * (1 - rate) + img_origin[y][x] * rate
	return img

def fillBlack(origin_img, new_img):
	# newImg = np.zeros(origin_img.shape)
	# newImg[:] = 255
	alpha = 0.5
	gray_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	mask = cv2.inRange(gray_img, 0, 0)
	white = mask[mask == 255].shape[0]
	total = mask.shape[0] * mask.shape[1]
	rate = white / float(total)
	# cv2.imwrite('mask.jpg', mask)
	h, w = gray_img.shape[0:2]
	if rate > 0.2:
		for y in xrange(h):
			for x in xrange(w):
				if mask[y][x] == 255:
					new_img[y][x] = new_img[y][x] * (1 - alpha) + origin_img[y][x] * alpha
					# new_img[y][x] = [255, 255, 255]
	return new_img

def fixRed(origin_img, new_img):
	alpha = 0.9
	maxRate = 1
	bgImg = cv2.imread('bg/black.jpg')
	mixed_clone = mixClone(origin_img, bgImg)
	mixed_clone = blue2red(mixed_clone)
	mixed_clone = histColor(mixed_clone)
	hsv = cv2.cvtColor(mixed_clone, cv2.COLOR_BGR2HSV)
	mask1 = cv2.inRange(hsv, (0, 43, 46), (10, 255, 255))
	mask2 = cv2.inRange(hsv, (156, 43, 46), (180, 255, 255))
	mask = mask1 + mask2
	h, w = mask.shape[0:2]
	rateList = []
	for y in xrange(h):
		for x in xrange(w):
			if mask[y][x] == 255:
				rate = hsv[y][x][2] / 255.0
				rateList.append(rate)
	maxRate = max(rateList)
	r = 1.0 / maxRate
	for y in xrange(h):
		for x in xrange(w):
			if mask[y][x] == 255:
				rate = hsv[y][x][2] / 255.0 * r
				new_img[y][x] = new_img[y][x] * (1 - rate) + origin_img[y][x] * rate
	return new_img

def StartCube(opt_sift, opt_obj, opt_bg, opt_mix, opt_alpha, opt_out):
	if opt_sift:
		taskId = '%s_'%random.randint(1000, 9999)
		trimap = makeSift(opt_obj, taskId)
		alphaname = matting_with_trimap(opt_obj, trimap, taskId)
		# for i in glob('tmp/%s*'%taskId):
		# 	os.remove(i)
		grayImg = cv2.imread(alphaname, 0)
		bgImg = cv2.imread(opt_bg)
		if opt_mix:
			objImg = cv2.imread(opt_obj)
			mixed_clone = mixClone(objImg, bgImg, float(opt_alpha))
			mixed_clone = blue2red(mixed_clone)
			mixed_clone = histColor(mixed_clone)
			mixImg = mixSift(mixed_clone, grayImg, objImg)
			# cv2.imwrite(opt_out + '_1', mixImg)
			# cv2.imwrite(opt_out + '_2', outImg)
			outImg = fixRed(objImg, mixImg)
			outImg = histYUV(outImg)
			outImg = fillBlack(objImg, outImg)
		else:
			outImg = mixClone(grayImg, bgImg, float(opt_alpha))
	else:
		objImg = cv2.imread(opt_obj)
		# objImg = cv2.imread('tmp/6474_zbnwl.jpg')
		bgImg = cv2.imread(opt_bg)
		mixed_clone = mixClone(objImg, bgImg, float(opt_alpha))
		mixed_clone = blue2red(mixed_clone)
		outImg = fixRed(objImg, mixed_clone)
		outImg = histYUV(outImg)
		outImg = fillBlack(objImg, outImg)
	cv2.imwrite(opt_out, outImg)


if __name__ == '__main__':
	USAGE = 'usage: python cube.py'
	parser = OptionParser(USAGE)
	parser.add_option('--bg', dest = 'bg', default = None)
	parser.add_option('--obj', dest = 'obj', default = None)
	parser.add_option('--out', dest = 'out', default = 'output.jpg')
	# parser.add_option('--style', dest = 'style', default = None)
	parser.add_option('--alpha', dest = 'alpha', default = 1.0)
	parser.add_option('--sift', action = 'store_true', dest = 'sift', default = False)
	parser.add_option('--mix', action = 'store_true', dest = 'mix', default = False)
	opt, args = parser.parse_args()
	StartCube(opt.sift, opt.obj, opt.bg, opt.mix, opt.alpha, opt.out)