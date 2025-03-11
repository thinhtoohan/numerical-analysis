import numpy as np
def gauss_seidel(A, b, x0=None, max_iter=300, tolerance=1e-2):
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()  # Create a copy to avoid modifying the original x0

    for iteration in range(max_iter):
        x_old = x.copy()  # Store the previous iteration's values
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x[j]
            x[i] = (b[i] - sigma) / A[i, i]

        if np.linalg.norm(x - x_old, ord=np.inf) < tolerance:
            print(f"converged after {iteration}")
            return x  # Converged

    return None # Did not converge

# Example usage:
# A = np.array([[4,1,1,0,1],[-1,-3,1,1,0],[2,1,5,-1,-1],[-1,-1,-1,4,0],[0,2,-1,1,4]])
# b = np.array([[6,6,6,6,6]]).T

# solution = gauss_seidel(A, b)

# A = np.array([[2,-1,1],[2,2,2],[-1,-1,2]])
# b = np.array([[-1,4,-5]]).T 

A = np.array([[1,0,-2],[-0.5,1,-0.25],[1,-0.5,1]])
b = np.array([[0.2,-1.425,2]]).T
solution = gauss_seidel(A, b)


if solution is not None:
    print("Solution:", solution)
else:
    print("Gauss-Seidel method did not converge within the specified iterations.")

