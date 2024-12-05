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
j_Aleft = 1
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


def apply_motion_blur(img, n, j_Aleft, k_Aleft, j_Aright, k_Aright, filename):
    """
    Apply motion blur to an image using two motion blurring matrices: Aleft and Aright.

    Parameters:
    - image: np array, the raw matrix for original image
    - n: int, the size of the image (assumes square image).
    - j_Aleft: int, parameter for Aleft matrix.
    - k_Aleft: int, parameter for Aleft matrix.
    - j_Aright: int, parameter for Aright matrix.
    - k_Aright: int, parameter for Aright matrix.

    Returns:
    - B: The blurred image.
    """

    img=np.copy(img)
    # Create Aleft matrix
    motion_blurring_Aleft = MotionTypeBlurringMatrix(n, k_Aleft, j_Aleft)
    Aleft = motion_blurring_Aleft.get_matrix()
    # an_of_Aleft = motion_blurring_Aleft.get_dictionary()
    # print(f"the an of Aleft for {filename} is: \n ")
    # print(an_of_Aleft)
    
    # Create Aright matrix
    motion_blurring_Aright = MotionTypeBlurringMatrix(n, k_Aright, j_Aright)
    Aright = motion_blurring_Aright.get_matrix()
    # an_of_Aright = motion_blurring_Aright.get_dictionary()
    # print(f"the an of Aright for {filename} is: \n ")
    # print(an_of_Aright)
    
    # Apply the motion blur: B = Aleft @ img @ Aright
    B = Aleft @ img @ Aright
    
    return B
    

def blur_all_png_files(source_image_dir, j_Aleft, k_Aleft, j_Aright, k_Aright, blurred_image_dir):
    for filename in os.listdir(source_image_dir):
        if filename.endswith(".png"): 
            print(f"Now I am blurring {filename}")
            source_image_path = os.path.join(source_image_dir, filename)
            output_filename = filename.replace("original", "blurred")
            output_image_path = os.path.join(blurred_image_dir, output_filename)
            im = Image.open(source_image_path) 
            img = np.array(im)
            img = img.astype(np.float64) / 255
            img = convert_RGB_RGBA_to_grayscale(img)
            m, n = img.shape #m=n
            print(img.shape)

            blurred_matrix = apply_motion_blur(img, n, j_Aleft, k_Aleft, j_Aright, k_Aright, filename)

            # Save the resulting image
            plt.figure(1)
            plt.axis('off')
            plt.gray()
            plt.imshow(blurred_matrix)
            plt.imsave(output_image_path, blurred_matrix)

blur_all_png_files(source_image_dir=source_image_dir, j_Aleft=j_Aleft, k_Aleft=k_Aleft, j_Aright=j_Aright, k_Aright=k_Aright, blurred_image_dir=blurred_image_dir)

def LU_decomposition_with_pivoting(matrix):
    P, L, U = lu(matrix)
    return P, L, U

def recover_matrix_from_blurred_matrix(blurred_matrix, n, j_Aleft, k_Aleft, j_Aright, k_Aright, filename):
    blurred_matrix=np.copy(blurred_matrix)

    # find Aleft matrix
    motion_blurring_Aleft = MotionTypeBlurringMatrix(n, k_Aleft, j_Aleft)
    Aleft = motion_blurring_Aleft.get_matrix()
    # an_of_Aleft = motion_blurring_Aleft.get_dictionary()
    # print(f"the an of Aleft for {filename} is: \n ")
    # print(an_of_Aleft)
    a_dict = motion_blurring_Aleft.get_dictionary()
    # 遍历字典并打印出键和值
    for key, value in a_dict.items():
        if value != 0:
            print(f'Aleft的an: 键: {key}, 值: {value}')
    
    # find Aright matrix
    motion_blurring_Aright = MotionTypeBlurringMatrix(n, k_Aright, j_Aright)
    Aright = motion_blurring_Aright.get_matrix()
    # an_of_Aright = motion_blurring_Aright.get_dictionary()
    # print(f"the an of Aright for {filename} is: \n ")
    # print(an_of_Aright)

    rank_of_Aleft=np.linalg.matrix_rank(Aleft)
    rank_of_Aright=np.linalg.matrix_rank(Aright)
    print(f"rank of aleft={rank_of_Aleft}")
    print(f"rank of aright={rank_of_Aright}")



    


    Pleft, Lleft, Uleft = LU_decomposition_with_pivoting(Aleft)
    Pright, Lright, Uright = LU_decomposition_with_pivoting(Aright)
    det_Uleft = np.linalg.det(Uleft)
    print(f"det(Uleft) = {det_Uleft}")
    if np.isclose(det_Uleft, 0):
        print("Uleft is singular.")


    Uleft_times_X_times_Aright = solve_triangular(Lleft, (Pleft @ blurred_matrix), lower=True)  # solve_triangular is a very accurate and stable package for solving Lx = b
    print(f"I am the Uleft for {filename}, and my det is {det_Uleft} ,and very soon will execute line 195")
    X_times_Aright = solve_triangular(Uleft, Uleft_times_X_times_Aright, lower=False)
    #  NOW we want to solve Uright.T * Lright.T * Pright * X.T = (X_times_Aright).T
    Lright_T_times_Pright_times_X_T = solve_triangular(Uright.T, (X_times_Aright).T, lower=True)
    Pright_times_X_T = solve_triangular(Lright.T,  Lright_T_times_Pright_times_X_T, lower=False)
    X_T = np.linalg.solve(Pright, Pright_times_X_T)
    X = X_T.T 
    return X

def recover_all_blurred_files(blurred_image_dir, j_Aleft, k_Aleft, j_Aright, k_Aright, recovered_image_dir):
    for filename in os.listdir(blurred_image_dir):
        if filename.endswith(".png"): 
            print(f"Now I am deblurring {filename}")
            blurred_image_path = os.path.join(blurred_image_dir, filename)
            output_filename = filename.replace("blurred", "recovered")
            output_image_path = os.path.join(recovered_image_dir, output_filename)
            im = Image.open(blurred_image_path) 
            img = np.array(im)
            img = img.astype(np.float64) / 255
            img = convert_RGB_RGBA_to_grayscale(img)
            m, n = img.shape #m=n
            print(img.shape)
            recovered_matrix = recover_matrix_from_blurred_matrix(img, n, j_Aleft, k_Aleft, j_Aright, k_Aright, filename)
            
            # Save the resulting image
            plt.figure(1)
            plt.axis('off')
            plt.gray()
            plt.imshow(recovered_matrix)
            plt.imsave(output_image_path, recovered_matrix)

recover_all_blurred_files(blurred_image_dir, j_Aleft, k_Aleft, j_Aright, k_Aright, recovered_image_dir)




