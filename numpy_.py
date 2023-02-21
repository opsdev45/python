import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# 1

# np.zeros((5,3)) 
# np.array([[0,1,2],[3,4,5]]) # list of list 
# np.arange(6)
# np.random.random((4,9))
# np.random.random_integers() 
# np.arange(6)
# x = np.random.random(3,4) 
# print(x)


# x = np.random.random((4,3))
# print(x)


# x = np.array([1, 2, 3])
# print(x)


# 2 
def rgb2gray(rgb):
    return np.round(np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]))


img = mpimg.imread('venomjpg.jpg')
print(img)
gray = rgb2gray(img)
old_mat = gray.copy()

gray[gray < np.percentile(gray,25)] = 0
gray[(gray < np.percentile(gray,25)) & (gray < np.percentile(gray,50))] = np.percentile(gray,50)
gray[(gray < np.percentile(gray,50)) & (gray < np.percentile(gray,75))] = np.percentile(gray,75)
gray[np.percentile(gray,75) < gray] = 255

# only_black_and_white = (gray > 160) * 255
plt.imshow(np.hstack((old_mat, gray)), cmap=plt.get_cmap('gray'))

# plt.imshow(gray, cmap=plt.get_cmap('gray'))
plt.show()

hist, bins = np.histogram(old_mat, bins=256, range=(0, 255))

# Plot the histograms
# plt.plot(hist)
plt.bar(bins[:-1], hist, width=3)
plt.xlabel('Gray Level')
plt.ylabel('Number of Pixels')
plt.show()
# plt.imshow(gray / 255, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)

# hist: An array of counts, where each element represents the number of pixels that fall into the corresponding bin.

# bins: An array of bin edges. The left edges of each bin are stored in the array, along with the right edge of the last bin.



# 3
v , m = np.unique(old_mat,return_index=True)
plt.bar(v,m)
plt.xlabel('Gray Level')
plt.ylabel('Number of Pixels')
plt.show()