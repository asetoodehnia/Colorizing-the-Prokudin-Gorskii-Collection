import numpy as np
import skimage as sk
import skimage.io as skio
from skimage.transform import rescale
from scipy.ndimage import sobel
import os


#####################
#### MAIN METHOD ####
#####################

def main():
	if not os.path.exists('results'):
		os.makedirs('results')

	for filename in os.listdir("data"):
		for method in ["SSD", "NCC"]:
			print(filename)
			###################################################
			#### using naive method on smaller .jpg images ####
			###################################################
			if filename.endswith(".jpg"): 
				b, g, r = split_image("data/" + filename)

				# align the green and red color channels to the blue color channel
				ag, g_shift = align(g, b)
				ar, r_shift = align(r, b)

				# stack the color channels
				im_out = np.dstack([ar, ag, b])
   
   				# save the image
				skio.imsave("results/" + str.split(filename, ".")[0] + "_" + method + "_g_" + str(g_shift) + \
				            "_r_" + str(r_shift) + ".jpg", im_out)

			####################################################
			#### using pyramid method on larger .tif images ####
			####################################################
			elif filename.endswith(".tif"):
				b, g, r = split_image("data/" + filename)

				# apply sobel edge filter (improvement is neglibile for all images other than emir.tiff)
				b_s = np.abs(sobel(b))
				g_s = np.abs(sobel(g))
				r_s = np.abs(sobel(r))

				_, g_shift = pyramid(g_s, b_s)
				_, r_shift = pyramid(r_s, b_s)

				# apply displacement shift to original color channels
				ag = np.roll(g, g_shift, (0,1))
				ar = np.roll(r, r_shift, (0,1))

				# stack the color channels
				im_out = np.dstack([ar, ag, b])

				# save the image
				skio.imsave("results/" + str.split(filename, ".")[0] + "_" + method + "_g_" + str(g_shift) + \
				            "_r_" + str(r_shift) + ".jpg", im_out)

##########################
#### HELPER FUNCTIONS ####
##########################

def split_image(file_name):
	'''
	splits the image at file_name into three color channels
	'''

	# read in the image
	im = skio.imread(file_name)

	# convert to double (might want to do this later on to save memory)    
	im = sk.img_as_float(im)

	# compute the height of each part (just 1/3 of total)
	height = np.floor(im.shape[0] / 3.0).astype(np.int)

	# separate color channels
	b = im[:height]
	g = im[height: 2*height]
	r = im[2*height: 3*height]

	return b, g, r

def align(im1, im2, method='SSD', off_x=(-15, 15), off_y=(-15, 15)):
	'''
	Does an exhaustive search over the range specified by
	off_x and off_y to find the optimal displacement vector 
	for im1 over im2
	'''
	# here I crop the images to ignore the borders
	im1_c = im1[int(0.1 * len(im1)):-int(0.1 * len(im1)), 
				int(0.1 * len(im1[0])):-int(0.1 * len(im1[0]))]
	im2_c = im2[int(0.1 * len(im2)):-int(0.1 * len(im2)), 
				int(0.1 * len(im2[0])):-int(0.1 * len(im2[0]))]

	best_score = -float('inf')
	best_shift = [0, 0]
	
	# loop over all the different displacement permutations
	for i in range(off_x[0], off_x[1] + 1):
		for j in range(off_y[0], off_y[1] + 1):
			temp_score = score(np.roll(im1_c, (i, j), (0, 1)), im2_c, method)
			if temp_score > best_score:
				best_score = temp_score
				best_shift = [i, j]

    # return the best displaced image along with the displacement vector
	return np.roll(im1, best_shift, (0, 1)), np.array(best_shift)

def score(im1, im2, method='SSD'):
	'''
	returns the similarity score of im1 and im2
	'''
	if method == 'SSD':
		return -np.sum(np.sum((im1 - im2)**2)) 
	elif method == 'NCC':
		im1 = np.ndarray.flatten(im1)
		im2 = np.ndarray.flatten(im2)
		return np.dot(im1 / np.linalg.norm(im1), im2 / np.linalg.norm(im2))

def pyramid(im1, im2, method='SSD', off_x=(-4, 4), off_y=(-4, 4), depth=5):
	'''
	uses the image pyramid method to deal with large-scale images
	'''
	# base case: either image width has been met or it's reached the max depth
	if im1.shape[0] < 400 or depth == 0:
		# call the naive align funciton
		return align(im1, im2, method)
	else:
		# recurse on a half scaled image
		_, best_shift = pyramid(rescale(im1, 0.5), rescale(im2, 0.5), method, depth=depth - 1)

		# scale the displacement shift vector back up
		best_shift *= 2

		# find the best shifted image and displacement vector at this level
		result, new_shift = align(np.roll(im1, best_shift, (0, 1)), im2, method, off_x, off_y)

		# add the new displacement shift vector to the best_shift vector and return
		best_shift += new_shift
		
		return result, best_shift

if __name__ == "__main__":
	main()
