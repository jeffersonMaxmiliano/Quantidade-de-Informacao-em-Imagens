import io
import urllib.request

# 3rd Party
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

def get_entropy(signal):
   """ Uses log2 as base
   """
   probabability_distribution = [np.size(signal[signal == i])/(1.0 * signal.size) for i in list(set(signal))]
   entropy = np.sum([pp * np.log2(1.0 / pp) for pp in probabability_distribution])
   return entropy
image_url = 'http://i.imgur.com/8vuLtqi.png' # Lena Image
fd = urllib.request.urlopen(image_url)
image_file = io.BytesIO(fd.read())
img_color = Image.open(image_file)
img_grey = img_color.convert('L')
img_color = np.array(img_color)
img_grey = np.array(img_grey)
neighborhood=5

def get_entropy_of_image(img_grey, nh):
   dim0, dim1 = img_grey.shape
   entropy=np.array(img_grey)
   for row in range(dim0):
       for col in range(dim1):
           lower_x = np.max([0, col - nh])
           upper_x = np.min([dim1, col + nh])
           lower_y = np.max([0, row - nh])
           upper_y = np.min([dim0, row + nh])
           area = img_grey[lower_y: upper_y, lower_x: upper_x].flatten()
           entropy[row, col] = get_entropy(area)
   return entropy
#link do material http://bugra.github.io/work/notes/2014-05-16/entropy-perplexity-image-text/
# Get the entropy of the image
img_entropy = get_entropy_of_image(img_grey, neighborhood)
plt.figure(figsize=(12, 12));
plt.imshow(img_color);
plt.grid(False);
plt.xticks([]);
plt.yticks([]);
plt.figure(figsize=(12, 12));
plt.imshow(img_grey, cmap=plt.cm.gray);
plt.grid(False);
plt.xticks([]);
plt.yticks([]);
plt.figure(figsize=(12, 12));
plt.imshow(img_entropy, cmap=plt.cm.Purples);
plt.title('Entropy in {}x{} neighbourhood'.format(2*neighborhood, 2*neighborhood));
plt.colorbar();
plt.grid(False);
plt.xticks([]);
plt.yticks([]);
# 10 x 10 neighbor
lowpass = ndimage.gaussian_filter(img_grey, 10)
# Subtracting the lowpass, we get the highpass
highpass = img_grey - lowpass
plt.figure(figsize=(12, 12));
plt.imshow(highpass, cmap=plt.cm.Purples);
plt.colorbar();
plt.grid(False);
plt.xticks([]);
plt.yticks([]);

