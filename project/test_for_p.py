import numpy as np
from PIL import Image
from numpy import asarray
from matplotlib import image
from matplotlib import pyplot as plt
import os
from scipy.linalg import qr, svd, lu_factor, lu_solve, solve_triangular
from scipy.sparse import csr_matrix

# Parameters for Aleft
j_Aleft = 2
k_Aleft = 5
# Parameters for Aright
j_Aright = 0
k_Aright = 20 # meaning the motion looks like moving to right

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


blurred_image_path=r'C:\dda3005\project\blurred_images\2048_mountain_blurred.png'
im = Image.open(blurred_image_path) 
img = np.array(im)
img = img.astype(np.float64) / 255
print("before convert_RGB_RGBA_to_grayscale", img.shape)
img = convert_RGB_RGBA_to_grayscale(img)
print("after convert_RGB_RGBA_to_grayscale", img.shape)
m, n = img.shape #m=n
blurred_matrix=np.copy(img)

# find Aleft matrix
motion_blurring_Aleft = MotionTypeBlurringMatrix(n, k_Aleft, j_Aleft)
Aleft = motion_blurring_Aleft.get_matrix()

# find Aright matrix
motion_blurring_Aright = MotionTypeBlurringMatrix(n, k_Aright, j_Aright)
Aright = motion_blurring_Aright.get_matrix()

Qleft, Rleft= qr(Aleft)
diagonal_elements = Rleft.diagonal()
# 定义接近零的阈值
threshold = 1e-5
# 筛选出接近零的元素
close_to_zero = diagonal_elements[np.abs(diagonal_elements) < threshold]
# 打印接近零的元素
print("接近零的元素:", close_to_zero)
print(diagonal_elements)
# 自定义分页显示函数
def paginate(iterable, page_size=20):
    for i in range(0, len(iterable), page_size):
        yield iterable[i:i+page_size]


# 分页显示对角线元素
for page in paginate(diagonal_elements):
    print(page)
    input("按回车键继续...")



# plt.figure(1)
# plt.axis('off')
# plt.gray()
# plt.imshow(img)
# plt.show()







# # Create a simple 2D grayscale image (e.g., 4x4)
# img = np.array([[0.1, 0.2, 0.3, 0.4],
#                 [0.5, 0.6, 0.7, 0.8],
#                 [0.9, 1.0, 1.1, 1.2],
#                 [1.3, 1.4, 1.5, 1.6]])

# print(f"Original shape: {img.shape}")
# print(img)


# # Convert to 3D by adding a new axis for the color channel
# img_1 = img[:, :, np.newaxis]
# print(f"shape: {img_1.shape}")
# print(img_1)


# img_2= np.repeat(img[:, :, np.newaxis], 3, axis=2)  
# print(f"shape: {img_2.shape}")
# print(img_2)

# img_3= np.repeat(img_1[:, :, np.newaxis], 3, axis=2) 
# print(f"shape: {img_3.shape}")
# print(img_3)

# img_4= np.repeat(img[:, np.newaxis], 3, axis=1) 
# print(f"shape: {img_4.shape}")
# print(img_4)

# img_5= img[:, np.newaxis]
# print(f"shape: {img_5.shape}")
# print(img_5)