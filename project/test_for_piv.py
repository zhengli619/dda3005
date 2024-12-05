import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os
from scipy.linalg import qr, svd, lu_factor, lu_solve, solve_triangular
from scipy.sparse import csr_matrix
import time

# Define directories
source_image_dir = r'C:\dda3005\project\source_images'
blurred_image_dir = r'C:\dda3005\project\blurred_images'
recovered_image_dir_LU = r'C:\dda3005\project\recovered_images_LU'
recovered_image_dir_QR = r'C:\dda3005\project\recovered_images_QR'

# Parameters for Aleft
j_Aleft = 1
k_Aleft = 5
# Parameters for Aright
j_Aright = 0
k_Aright = 20  # Meaning the motion looks like moving to the right

# Helper function to convert RGB/RGBA to grayscale
def convert_RGB_RGBA_to_grayscale(img):
    """
    Convert an RGB or RGBA image to grayscale.
    
    Parameters:
    - img: np array, image data
    
    Returns:
    - img_gray: np array, grayscale image
    """
    img = np.copy(img)
    if len(img.shape) == 3 and img.shape[2] in [3, 4]:
        if img.shape[2] == 4:
            img_rgb = img[:, :, :3]
        else:
            img_rgb = img
        img_gray = 0.2989 * img_rgb[:, :, 0] + 0.5870 * img_rgb[:, :, 1] + 0.1140 * img_rgb[:, :, 2]
        return img_gray
    else:
        return img  # Already grayscale

# Helper function to make matrix non-singular using SVD
def make_matrix_non_singular(A, epsilon=1e-6):
    """
    Make matrix A invertible by modifying its singular values.
    
    Parameters:
    - A: the input matrix (numpy.ndarray)
    - epsilon: threshold
    
    Returns:
    - A_non_singular: the output invertible matrix (numpy.ndarray)
    """
    print("Performing SVD decomposition...")
    U, S, VT = svd(A, full_matrices=False)
    print("SVD decomposition completed.")

    # Modify singular values: if < epsilon, set to epsilon
    S_adjusted = np.where(S < epsilon, epsilon, S)

    # Construct new diagonal matrix S'
    Sigma_adjusted = np.diag(S_adjusted)

    # Construct A' = U * S' * V^T
    # print("Constructing A' (modified matrix)...")
    A_non_singular = U @ Sigma_adjusted @ VT
    # print("Construction of A' completed.")

    return A_non_singular

# Class to create motion type blurring matrices
class MotionTypeBlurringMatrix:
    def __init__(self, n, k, j=0):
        """
        Initialize the motion type blurring matrix.
        
        Parameters:
        - n: Dimension of the matrix (n x n)
        - k: Parameter k
        - j: Parameter j, default is 0
        """
        self.n = n
        self.k = k
        self.j = j
        self.blurring_matrix = np.zeros((n, n))
        self._initialize_blurring_matrix()

    def _initialize_blurring_matrix(self):
        """
        Initialize the blurring matrix based on motion parameters.
        """
        a_values = self._compute_a_values()
        # Fill the upper triangle
        for i in range(1, self.n + 1):
            for j_idx in range(1, i + 1):
                key = f'a_{i}'
                self.blurring_matrix[j_idx - 1, self.n - i + j_idx - 1] = a_values.get(key, 0)
        # Fill the lower triangle
        for i in range(self.n + 1, 2 * self.n):
            for j_idx in range(1, 2 * self.n + 1 - i):
                key = f'a_{i}'
                self.blurring_matrix[i - self.n + j_idx - 1, j_idx - 1] = a_values.get(key, 0)

    def _compute_a_values(self):
        """
        Compute the a_i values for the blurring matrix.

        Returns:
        - a_dict: Dictionary of a_i values
        """
        a_dict = {f'a_{i}': 0 for i in range(1, 2 * self.n)}
        for i in range(0, self.k):
            key = f'a_{self.n + self.j - i}'
            a_dict[key] = (2 / (self.k * (self.k + 1))) * (self.k - i)
        return a_dict

    def get_matrix(self):
        """
        Retrieve the blurring matrix.

        Returns:
        - blurring_matrix: np array, blurring matrix
        """
        return self.blurring_matrix

# Function to apply motion blur to an image
def apply_motion_blur(img, n, j_Aleft, k_Aleft, j_Aright, k_Aright):
    """
    Apply motion blur to an image using Aleft and Aright matrices.

    Parameters:
    - img: np array, grayscale image
    - n: int, size of the image (assuming square)
    - j_Aleft, k_Aleft: Parameters for Aleft matrix
    - j_Aright, k_Aright: Parameters for Aright matrix

    Returns:
    - B: np array, blurred image
    """
    motion_blurring_Aleft = MotionTypeBlurringMatrix(n, k_Aleft, j_Aleft)
    Aleft = motion_blurring_Aleft.get_matrix()

    motion_blurring_Aright = MotionTypeBlurringMatrix(n, k_Aright, j_Aright)
    Aright = motion_blurring_Aright.get_matrix()

    B = Aleft @ img @ Aright
    return B

# Function to blur all PNG files in the source directory
def blur_all_png_files(source_dir, j_Aleft, k_Aleft, j_Aright, k_Aright, blurred_dir):
    """
    Apply motion blur to all PNG images in the source directory.

    Parameters:
    - source_dir: str, path to source images
    - j_Aleft, k_Aleft: Parameters for Aleft matrix
    - j_Aright, k_Aright: Parameters for Aright matrix
    - blurred_dir: str, path to save blurred images
    """
    if not os.path.exists(blurred_dir):
        os.makedirs(blurred_dir)
        print(f"Created directory for blurred images: {blurred_dir}")

    for filename in os.listdir(source_dir):
        if filename.lower().endswith(".png"):
            print(f"Blurring {filename}...")
            source_path = os.path.join(source_dir, filename)
            output_filename = filename.replace("original", "blurred")
            output_path = os.path.join(blurred_dir, output_filename)

            img = Image.open(source_path)
            img = np.array(img).astype(np.float64) / 255.0
            img_gray = convert_RGB_RGBA_to_grayscale(img)
            m, n = img_gray.shape  # Assuming square images

            blurred_img = apply_motion_blur(img_gray, n, j_Aleft, k_Aleft, j_Aright, k_Aright)

            # Save the resulting image
            plt.imsave(output_path, blurred_img, cmap='gray')
            print(f"Blurred image saved to {output_path}")

# Function to recover the original image using LU decomposition
def recover_matrix_from_blurred_matrix_LU(blurred_matrix, n, j_Aleft, k_Aleft, j_Aright, k_Aright):
    """
    Recover the original image matrix from the blurred matrix using LU decomposition.

    Parameters:
    - blurred_matrix: np array, blurred image
    - n: int, size of the image (assuming square)
    - j_Aleft, k_Aleft: Parameters for Aleft matrix
    - j_Aright, k_Aright: Parameters for Aright matrix

    Returns:
    - X: np array, recovered image
    """
    # Construct Aleft and Aright matrices
    motion_blurring_Aleft = MotionTypeBlurringMatrix(n, k_Aleft, j_Aleft)
    Aleft = motion_blurring_Aleft.get_matrix()

    motion_blurring_Aright = MotionTypeBlurringMatrix(n, k_Aright, j_Aright)
    Aright = motion_blurring_Aright.get_matrix()

    # Invert Aright
    Aright_inv = np.linalg.inv(Aright)

    # Transform B
    B_transformed = blurred_matrix @ Aright_inv

    # Start CPU timer
    start_time = time.time()

    # Make Aleft invertible
    Aleft_reg = make_matrix_non_singular(Aleft, epsilon=1e-6)

    # Solve Aleft * X = B_transformed using LU decomposition
    lu, piv = lu_factor(Aleft_reg)
    X = lu_solve((lu, piv), B_transformed)

    # End CPU timer
    end_time = time.time()
    cpu_time = end_time - start_time

    # Clip values of X to be in the range [0, 1]
    X = np.clip(X, 0, 1)

    return X, cpu_time

# Function to recover all blurred files using LU
def recover_all_blurred_files_LU(blurred_dir, j_Aleft, k_Aleft, j_Aright, k_Aright, recovered_dir, source_dir):
    """
    Recover all blurred PNG images using LU-based least squares.

    Parameters:
    - blurred_dir: str, path to blurred images
    - j_Aleft, k_Aleft: Parameters for Aleft matrix
    - j_Aright, k_Aright: Parameters for Aright matrix
    - recovered_dir: str, path to save recovered images
    - source_dir: str, path to original images for error calculation
    """
    if not os.path.exists(recovered_dir):
        os.makedirs(recovered_dir)
        print(f"Created directory for LU recovered images: {recovered_dir}")

    for filename in os.listdir(blurred_dir):
        if filename.lower().endswith(".png"):
            print(f"\nDeblurring {filename} using LU decomposition...")
            blurred_path = os.path.join(blurred_dir, filename)
            output_filename = filename.replace("blurred", "recovered_LU")
            output_path = os.path.join(recovered_dir, output_filename)

            # Load blurred image
            blurred_img = Image.open(blurred_path)
            blurred_img = np.array(blurred_img).astype(np.float64) / 255.0
            blurred_img_gray = convert_RGB_RGBA_to_grayscale(blurred_img)
            m, n = blurred_img_gray.shape  # Assuming square images

            # Recover the image and measure CPU time
            recovered_img, cpu_time = recover_matrix_from_blurred_matrix_LU(
                blurred_img_gray, n, j_Aleft, k_Aleft, j_Aright, k_Aright
            )
            print(f"Recovered image using LU in {cpu_time:.4f} seconds.")

            # Save the recovered image
            plt.imsave(output_path, recovered_img, cmap='gray')
            print(f"Recovered image saved to {output_path}")

            # Load the original image for error calculation
            original_filename = filename.replace("blurred", "original")
            original_path = os.path.join(source_dir, original_filename)
            if os.path.exists(original_path):
                original_img = Image.open(original_path)
                original_img = np.array(original_img).astype(np.float64) / 255.0
                original_img_gray = convert_RGB_RGBA_to_grayscale(original_img)

                # Compute relative forward error using Frobenius norm
                error_norm = np.linalg.norm(recovered_img - original_img_gray, 'fro')
                original_norm = np.linalg.norm(original_img_gray, 'fro')
                relative_error = error_norm / original_norm
                print(f"Relative Forward Error (LU): {relative_error:.6f}")
            else:
                print(f"Original image {original_filename} not found. Cannot compute error.")

# # Function to recover the original image using QR decomposition
# def recover_matrix_from_blurred_matrix_QR(blurred_matrix, n, j_Aleft, k_Aleft, j_Aright, k_Aright):
#     """
#     Recover the original image matrix from the blurred matrix using QR decomposition and pseudoinversion.

#     Parameters:
#     - blurred_matrix: np array, blurred image
#     - n: int, size of the image (assuming square)
#     - j_Aleft, k_Aleft: Parameters for Aleft matrix
#     - j_Aright, k_Aright: Parameters for Aright matrix

#     Returns:
#     - X: np array, recovered image
#     - cpu_time: float, time taken in seconds
#     """
#     # Construct Aleft and Aright matrices
#     motion_blurring_Aleft = MotionTypeBlurringMatrix(n, k_Aleft, j_Aleft)
#     Aleft = motion_blurring_Aleft.get_matrix()

#     motion_blurring_Aright = MotionTypeBlurringMatrix(n, k_Aright, j_Aright)
#     Aright = motion_blurring_Aright.get_matrix()

#     # Invert Aright
#     Aright_inv = np.linalg.inv(Aright)

#     # Transform the blurred image
#     B = blurred_matrix @ Aright_inv

#     # Start CPU timer
#     start_time = time.time()

#     # Perform QR decomposition on Aleft
#     Qleft, Rleft = qr(Aleft, mode='economic')

#     # Compute pseudoinverse of Rleft
#     Rleft_pinv = np.linalg.pinv(Rleft)

#     # Recover X using the formula: X = Rleft_pinv @ Qleft.T @ B
#     X = Rleft_pinv @ Qleft.T @ B

#     # End CPU timer
#     end_time = time.time()
#     cpu_time = end_time - start_time

#     # Clip the values to [0, 1] to maintain valid image intensities
#     X = np.clip(X, 0, 1)

#     return X, cpu_time

def pre_calculate(A, tol=1e-10):
    """
    Perform QR decomposition with column pivoting and determine the rank.
    
    Parameters:
    - A: Input matrix of shape (n, n)
    - tol: Tolerance for rank determination
    
    Returns:
    - Q: Orthogonal matrix from QR decomposition
    - R1: Upper triangular matrix corresponding to the rank
    - r: Rank of the matrix
    """
    # QR decomposition with pivoting
    # print("doing QR decomposition with pivoting ...")
    Q, R, P = qr(A, pivoting=True, mode='full') # Q: (m, m) R: (m, n)
    # print("QR decomposition done")
    print("the pivoting array is:", P)
    print(end="\n \n \n")

    # get rank of matrix
    diag_R = np.abs(np.diag(R))
    r = np.sum(diag_R > tol)
    # print(f"the rank of matrix is: {r}")

    # get R1 and c1
    R1 = R[:r, :r]  

    return Q, R1, r


def least_squares_solution_after_precalculation(Q, R1, r, b, tol=1e-10):
    """
    using QR decomposition to solve least square problem, fit for rank deficiency
    according to the formula of lecture slides to construct solution: X = P @ [R_inverse @ c1, 0, ..., 0]^T 
    in my project, I find P is just Identity

    parameter:
    - Q: Orthogonal matrix from QR decomposition: (m, n)
    - R1: Upper triangular matrix corresponding to the rank of A: (r,r)
    - r: Rank of the matrix A
    - b: target vector, (m,1)
    - tol: default: 1e-10

    return:
    - x: solution to least square problem: (n,1)
    """

    # compute Q^T b
    Qt_b = np.dot(Q.T, b)
    # compute c1
    c1 = Qt_b[:r]

    # solve R1 @ x1 = c1 to get R1_Inverse * c1
    x1 = solve_triangular(R1, c1)
    
    # construct [x1, 0, ..., 0]^T
    x_temp = np.zeros(Q.shape[1])
    x_temp[:r] = x1

    # I already know the P is I
    x = (x_temp)
    return x

def solve_AX_equals_B_using_qr_pivoting(A, B, tol=1e-10):
    """
    use least_squares_solution_after_precalculation to solve AX = B。
    
    parameter:
    - A:  (m, n)
    - B: target matrix: (n, n)
    - tol: defalut: 1e-10
    
    return:
    - X: solution of AX=B: (n, n)
    """
    n = A.shape[1]
    X = np.zeros_like(B)  #initialization
    
    Q, R1, r = pre_calculate(A, tol=1e-10)

    for i in range(n):
        b = B[:, i]
        x = least_squares_solution_after_precalculation(Q, R1, r, b, tol=1e-10)
        X[:, i] = x  # put it into the i th column of X
    
    return X


# Function to recover the original image using QR decomposition
def recover_matrix_from_blurred_matrix_QR(blurred_matrix, n, j_Aleft, k_Aleft, j_Aright, k_Aright):
    """
    Recover the original image matrix from the blurred matrix using QR decomposition and pseudoinversion.

    Parameters:
    - blurred_matrix: np array, blurred image
    - n: int, size of the image (assuming square)
    - j_Aleft, k_Aleft: Parameters for Aleft matrix
    - j_Aright, k_Aright: Parameters for Aright matrix

    Returns:
    - X: np array, recovered image
    - cpu_time: float, time taken in seconds
    """
    # Construct Aleft and Aright matrices
    motion_blurring_Aleft = MotionTypeBlurringMatrix(n, k_Aleft, j_Aleft)
    Aleft = motion_blurring_Aleft.get_matrix()

    motion_blurring_Aright = MotionTypeBlurringMatrix(n, k_Aright, j_Aright)
    Aright = motion_blurring_Aright.get_matrix()

    # Invert Aright
    Aright_inv = np.linalg.inv(Aright)

    # Transform the blurred image
    B = blurred_matrix @ Aright_inv

    # Start CPU timer
    start_time = time.time()

    # # Perform QR decomposition on Aleft
    # Qleft, Rleft = qr(Aleft, mode='economic')

    # # Compute pseudoinverse of Rleft
    # Rleft_pinv = np.linalg.pinv(Rleft)

    # # Recover X using the formula: X = Rleft_pinv @ Qleft.T @ B
    # X = Rleft_pinv @ Qleft.T @ B

    X= solve_AX_equals_B_using_qr_pivoting(Aleft, B, tol=1e-10)

    # End CPU timer
    end_time = time.time()
    cpu_time = end_time - start_time

    # Clip the values to [0, 1] to maintain valid image intensities
    X = np.clip(X, 0, 1)

    return X, cpu_time




# Function to recover all blurred files using QR
def recover_all_blurred_files_QR(blurred_dir, j_Aleft, k_Aleft, j_Aright, k_Aright, recovered_dir, source_dir):
    """
    Recover all blurred PNG images using QR-based least squares.

    Parameters:
    - blurred_dir: str, path to blurred images
    - j_Aleft, k_Aleft: Parameters for Aleft matrix
    - j_Aright, k_Aright: Parameters for Aright matrix
    - recovered_dir: str, path to save recovered images
    - source_dir: str, path to original images for error calculation
    """
    if not os.path.exists(recovered_dir):
        os.makedirs(recovered_dir)
        print(f"Created directory for QR recovered images: {recovered_dir}")

    for filename in os.listdir(blurred_dir):
        if filename.lower().endswith(".png"):
            print(f"\nDeblurring {filename} using QR decomposition...")
            blurred_path = os.path.join(blurred_dir, filename)
            output_filename = filename.replace("blurred", "recovered_QR")
            output_path = os.path.join(recovered_dir, output_filename)

            # Load blurred image
            blurred_img = Image.open(blurred_path)
            blurred_img = np.array(blurred_img).astype(np.float64) / 255.0
            blurred_img_gray = convert_RGB_RGBA_to_grayscale(blurred_img)
            m, n = blurred_img_gray.shape  # Assuming square images

            # Recover the image and measure CPU time
            recovered_img, cpu_time = recover_matrix_from_blurred_matrix_QR(
                blurred_img_gray, n, j_Aleft, k_Aleft, j_Aright, k_Aright
            )
            print(f"Recovered image using QR in {cpu_time:.4f} seconds.")

            # Save the recovered image
            plt.imsave(output_path, recovered_img, cmap='gray')
            print(f"Recovered image saved to {output_path}")

            # Load the original image for error calculation
            original_filename = filename.replace("blurred", "original")
            original_path = os.path.join(source_image_dir, original_filename)
            if os.path.exists(original_path):
                original_img = Image.open(original_path)
                original_img = np.array(original_img).astype(np.float64) / 255.0
                original_img_gray = convert_RGB_RGBA_to_grayscale(original_img)

                # Compute relative forward error using Frobenius norm
                error_norm = np.linalg.norm(recovered_img - original_img_gray, 'fro')
                original_norm = np.linalg.norm(original_img_gray, 'fro')
                relative_error = error_norm / original_norm
                print(f"Relative Forward Error (QR): {relative_error:.6f}")
            else:
                print(f"Original image {original_filename} not found. Cannot compute error.")



# Main execution block
def main():
    # Step 1: Apply motion blur to all original images
    print("Starting blurring process...")
    blur_all_png_files(
        source_dir=source_image_dir,
        j_Aleft=j_Aleft,
        k_Aleft=k_Aleft,
        j_Aright=j_Aright,
        k_Aright=k_Aright,
        blurred_dir=blurred_image_dir
    )
    print("Blurring process completed.")

    # Step 2: Recover blurred images using LU decomposition
    print("\nStarting recovery using LU decomposition...")
    recover_all_blurred_files_LU(
        blurred_dir=blurred_image_dir,
        j_Aleft=j_Aleft,
        k_Aleft=k_Aleft,
        j_Aright=j_Aright,
        k_Aright=k_Aright,
        recovered_dir=recovered_image_dir_LU,
        source_dir=source_image_dir
    )
    print("Recovery using LU decomposition completed.")

    # Step 3: Recover blurred images using QR decomposition
    print("\nStarting recovery using QR decomposition...")
    recover_all_blurred_files_QR(
        blurred_dir=blurred_image_dir,
        j_Aleft=j_Aleft,
        k_Aleft=k_Aleft,
        j_Aright=j_Aright,
        k_Aright=k_Aright,
        recovered_dir=recovered_image_dir_QR,
        source_dir=source_image_dir
    )
    print("Recovery using QR decomposition completed.")

    # Step 4: Summary of Results
    print("\nAll processes completed. Please check the recovered images and the console for CPU times and relative errors.")

if __name__ == "__main__":
    main()