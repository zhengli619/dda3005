import numpy as np
import matplotlib.pyplot as plt

def plot_polynomial(coefficients, title, x_range=(-50, 500), y_limit=None):
    x = np.linspace(x_range[0], x_range[1], 10000)
    y = np.polyval(coefficients, x)
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="Polynomial")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('p(x)')
    if y_limit is not None:
        plt.ylim(y_limit)  # set y axis area
    plt.show()
    

def companion_matrix(coefficients):
    # Ensure coefficients are in the correct form for the companion matrix
    # coefficients[0] should be the coefficient of the highest power
    coefficients = np.array(coefficients)
    n = len(coefficients) - 1  # Degree of the polynomial
    
    # Create an n x n companion matrix
    C = np.zeros((n, n))
    
    # Set the sub-diagonal elements to 1
    for i in range(1, n):
        C[i, i - 1] = 1
    
    # Set the last column to the negated coefficients (except the highest power)
    C[:, -1] = -coefficients[1:][::-1]
    
    return C

# Function to implement the algorithm
def qr_algorithm_with_rayleigh_quotient_shift(A, max_iter=100000, tol=1e-12):
    eigenvalues=[]

    #initialize
    X = A.copy()
    lambda_k_minus_1 = X[-1, -1] 
    n = A.shape[0]   

    for i in range(n,0,-1):
        print(f"now the matrix size is {i}")
        if i==1:
            print(f"the last eigenvalue is immediately found\n")
            eigenvalues.append(X[0,0]) 
            break 
        for k in range(max_iter):
            # QR decomposition
            shift_matrix = lambda_k_minus_1 * np.eye(i)
            X_shifted = X[:i, :i] - shift_matrix
            Q, R = np.linalg.qr(X_shifted)
            
            # Update X
            X[:i, :i] = R @ Q + shift_matrix
            # Pick a shift (eigenvalue approximation) - rayleigh quotient shift
            lambda_k_minus_1 = X[i-1, i-1]

            # Check the tolerance for convergence (norm of first n-1 entries in the last row)
            if np.linalg.norm(X[i-1, :i-1]) < tol:
                # Extract the converged eigenvalue
                eigenvalue = X[i-1, i-1]
                eigenvalues.append(eigenvalue)
                print(f"{i}th eigenvalue is found at iteration {k+1}\n")
                X=X[:i-1, :i-1]
                lambda_k_minus_1 = X[-1,-1]
                break
        # If we reach max_iter without convergence
        if k == max_iter-1:
            raise RuntimeError(f"Max iterations reached for {i}x{i} sub-matrix without convergence")

    return eigenvalues


def qr_algorithm_with_wilkinson_shift(A, max_iter=100000, tol=1e-12):
    """
    QR algorithm with Wilkinson shift to find eigenvalues of a matrix.
    """
    eigenvalues = []
    X = A.copy().astype(complex)
    n = X.shape[0]

    for i in range(n, 0, -1):
        print(f"Now processing {i}x{i} sub-matrix.")
        if i == 1:
            # The last remaining eigenvalue
            eigenvalues.append(X[0, 0])
            print("The last eigenvalue is immediately found.\n")
            break

        for k in range(max_iter):
            # Compute Wilkinson shift for the bottom-right 2x2 block
            if i > 1:
                submatrix_2x2 = X[i-2:i, i-2:i]
                eigenvalues_2x2 = np.linalg.eigvals(submatrix_2x2)
                lambda_k_minus_1 = eigenvalues_2x2[np.argmin(np.abs(eigenvalues_2x2 - submatrix_2x2[1, 1]))]
            else:
                lambda_k_minus_1 = X[-1, -1]

            # Apply the shift and perform QR decomposition
            shift_matrix = lambda_k_minus_1 * np.eye(i)
            X_shifted = X[:i, :i] - shift_matrix
            Q, R = np.linalg.qr(X_shifted)

            # Update matrix
            X[:i, :i] = R @ Q + shift_matrix

            # Check convergence
            if np.linalg.norm(X[i-1, :i-1]) < tol:
                eigenvalue = X[i-1, i-1]
                eigenvalues.append(eigenvalue)
                print(f"{i}th eigenvalue found at iteration {k+1}.")
                print(f"here is the convergence matrix for{i}*{i} size: ")
                print(np.array2string(X, precision=4, suppress_small=True)+"\n")
                # Deflate the matrix
                X = X[:i-1, :i-1]
                break

        else:
            # Max iterations reached
            raise RuntimeError(f"Max iterations reached for {i}x{i} sub-matrix without convergence.")

    return eigenvalues


# p(x) = x^4 - 324x^3 + 7175x^2 + 8100x - 180000
coefficients_p = [1, -324, 7175, 8100, -180000]
C_p = companion_matrix(coefficients_p)
print("Companion matrix of p(x):")
print(C_p)
print()
eigenvalues_of_p=qr_algorithm_with_rayleigh_quotient_shift(C_p, max_iter=100000, tol=1e-12)
eigenvalues_of_p = sorted(eigenvalues_of_p, key=lambda x: x.real)
print("Roots of p(x):")
for idx, root in enumerate(eigenvalues_of_p, start=1):
    print(f"Root {idx}: {root:.6f}")
print()

# q(x) = x^10 - 167x^9 + 10081x^8 - 251447x^7 + 1676815x^6 + 17367175x^5 - 66421125x^4 - 352378125x^3 + 454612500x^2 + 1949062500x
coefficients_q = [1, -167, 10081, -251447, 1676815, 17367175, -66421125, -352378125, 454612500, 1949062500, 0]
C_q = companion_matrix(coefficients_q)
print("\nCompanion matrix of q(x):")
print(C_q)
print()
eigenvalues_of_q=qr_algorithm_with_rayleigh_quotient_shift(C_q,max_iter=100000, tol=1e-12)
eigenvalues_of_q = sorted(eigenvalues_of_q, key=lambda x: x.real)
print("Roots of q(x):")
for idx, root in enumerate(eigenvalues_of_q, start=1):
    print(f"Root {idx}: {root:.6f}")
print()



#for question(c): What happens if the coefficients a2, a1, and a0 in the polynomial p are changed to a2 = 7225, a1 = −8100, and a0 = 180000?
coefficients_p_modified= [1, -324, 7225, -8100, 180000]
C_p_modified = companion_matrix(coefficients_p_modified)
print("Companion matrix of modified_p(x):")
print(C_p_modified)
# eigenvalues_of_p_modified=qr_algorithm_with_rayleigh_quotient_shift(C_p_modified, max_iter=100000, tol=1e-12)
# eigenvalues_of_p_modified = sorted(eigenvalues_of_p_modified, key=lambda x: x.real)
# print("Roots of p(x):")
# for idx, root in enumerate(eigenvalues_of_p_modified, start=1):
#     print(f"Root {idx}: {root:.6f}")
# print()
print()
plot_polynomial(coefficients_p, "Original Polynomial p(x)", y_limit=(-0.01, 0.01))
plot_polynomial(coefficients_p_modified, "Modified Polynomial p(x)", y_limit=(-0.01, 0.01))
print("we can find from graph that modified polynomial p(x) only has two real roots, that means there are complex roots there, \n\
so qr algorithm with rayleigh quotient shift can not handle it, thus, I try to use wilkinson shift.\n")
eigenvalues_of_p_modified=qr_algorithm_with_wilkinson_shift(C_p_modified, max_iter=100000, tol=1e-12)
eigenvalues_of_p_modified = sorted(eigenvalues_of_p_modified, key=lambda x: x.real)
print("Roots of modified_p(x):")
for idx, root in enumerate(eigenvalues_of_p_modified, start=1):
    print(f"Root {idx}: {root:.6f}")
print()
