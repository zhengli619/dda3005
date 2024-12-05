import numpy as np
from PIL import Image
from numpy import asarray
from matplotlib import image
from matplotlib import pyplot as plt
import os
from scipy.linalg import lu
from scipy.linalg import solve_triangular

source_image_dir= r'C:\dda3005\project\source_images'
blurred_image_dir = r'C:\dda3005\project\blurred_images'
recovered_image_dir = r'C:\dda3005\project\recovered_images'
# Parameters for Aleft
j_Aleft = 2
k_Aleft = 5
# Parameters for Aright
j_Aright = 0
k_Aright = 20 # meaning the motion looks like moving to right


def convert_RGB_RGBA_to_grayscale(img):
    img = np.copy(img)
    # Check if the image has 3 or 4 channels (RGB or RGBA)
    if len(img.shape) == 3 and img.shape[2] in [3, 4]:
        # Image is RGB or RGBA, so convert it to grayscale
        if img.shape[2] == 4:  # RGBA image, we need to remove the alpha channel
            img_rgb = img[:, :, :3]  # Take only the RGB channels
        else:
            img_rgb = img  # It's already an RGB image

        # Convert to grayscale using the weighted average
        img_gray = 0.2989 * img_rgb[:, :, 0] + 0.5870 * img_rgb[:, :, 1] + 0.1140 * img_rgb[:, :, 2]
        return img_gray
    else:
        # Image is already grayscale
        img_gray = img  # No conversion needed
        return img_gray


class MotionTypeBlurringMatrix:
    def __init__(self, n, k, j=0):
        """
        Initialize the parameters for the motion type blurring matrix.
        
        :param n: Dimension of the matrix (n x n)
        :param k: Parameter k
        :param j: Parameter j, default is 0
        """
        self.n = n
        self.k = k
        self.j = j
        self.a_dictionary_for_blurring_matrix = {}  # Dictionary to store a_i values
        self.blurring_matrix = np.zeros((n, n))  # Initialize the matrix as a zero matrix
        
        # Initialize the a_i values in the dictionary
        self._initialize_a_dictionary_for_blurring_matrix()
        
        # Create the motion type blurring matrix based on the a_i values
        self._create_blurring_matrix()

    def _initialize_a_dictionary_for_blurring_matrix(self):
        """
        Initialize the dictionary a_i where a_i are the weight values needed for the matrix.
        """
        for i in range(1, 2 * self.n):
            self.a_dictionary_for_blurring_matrix[f'a_{i}'] = 0  # Initialize to 0 by default
        for i in range(0, self.k):
            self.a_dictionary_for_blurring_matrix[f'a_{self.n + self.j - i}'] = (2 / (self.k * (self.k + 1))) * (self.k - i)

    def _create_blurring_matrix(self):
        """
        Fill the motion type blurring matrix based on the initialized dictionary a_i.
        """
        # Fill the upper triangle part of the matrix (n x n)
        for i in range(1, self.n + 1):
            for j in range(1, i + 1):
                self.blurring_matrix[j - 1, self.n - i + j - 1] = self.a_dictionary_for_blurring_matrix[f'a_{i}']
        
        # Fill the lower triangle part of the matrix (n x n)
        for i in range(self.n + 1, 2 * self.n):
            for j in range(1, 2 * self.n + 1 - i):
                self.blurring_matrix[i - self.n + j - 1, j - 1] = self.a_dictionary_for_blurring_matrix[f'a_{i}']
    
    def get_matrix(self):
        """
        Return the final motion type blurring matrix.
        """
        return self.blurring_matrix
    
    def get_dictionary(self):
        """
        Return the dictionary containing the a_i values used to generate the matrix.
        """
        return self.a_dictionary_for_blurring_matrix


source_image_path=r'C:\dda3005\project\source_images\3500_garlic_original.png'
im = Image.open(source_image_path) 
img = np.array(im)
# print(img)
# plt.figure(1)
# plt.axis('off')
# plt.gray()
# plt.imshow(img)
# plt.show()
img = img.astype(np.float64) / 255
plt.figure(1)
plt.axis('off')
plt.gray()
plt.imshow(img)
plt.show()
print(img.shape)
m,n= img.shape
motion_blurring_Aleft = MotionTypeBlurringMatrix(n, k_Aleft, j_Aleft)
Aleft = motion_blurring_Aleft.get_matrix()
a_dict = motion_blurring_Aleft.get_dictionary()
# 遍历字典并打印出键和值
for key, value in a_dict.items():
    if value != 0:
        print(f'键: {key}, 值: {value}')
