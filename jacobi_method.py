import numpy as np
from numpy.linalg import inv
class matrix:
    def __init__(self,matrix:np.array,b:np.array):
        self.matrix = matrix
        self.b = b
        self.n = matrix.shape[0]
        self.epsilon = 1e-3
    def jacobi_method(self):
        self.L = np.zeros((self.n, self.n))
        self.U = np.zeros((self.n, self.n))
        self.D = np.zeros((self.n, self.n))
        di = np.diag_indices(self.n)
        self.D[di] = self.matrix[di]
        print("D")
        print(self.D)
        for i in range(1, self.n):  # Start from the second row (index 1)
            for j in range(i):  # Iterate up to the diagonal element (exclusive)
                self.L[i][j] = self.matrix[i][j]
        print("L")
        self.L = -1*self.L
        print(self.L)
        for i in range(self.n - 1):  # Iterate up to the second-to-last row
            for j in range(i + 1, self.n):  # Iterate from the element after the diagonal
                self.U[i][j] = self.matrix[i][j]
        self.U = -1*self.U
        print("U")
        print(self.U)
        self.guess = np.zeros(self.n).reshape(self.n, 1)
        self.T = inv(self.D)@(self.L+self.U)
        print("T")
        print(self.T)
        eigvals = np.linalg.eigvals(self.T)
        spectral_radius = max(abs(eigvals))
        print(f"Spectral Radius of T: {spectral_radius}")
        self.infinity_norm_T = np.around(np.linalg.norm(self.T, np.inf),4)
        print(f"Infinity Norm of T: {self.infinity_norm_T}")
        self.c = inv(self.D)@self.b
        print("c")
        print(self.c)
        for i in range(0,25):
            self.guess = (self.T@self.guess)+self.c
            #print(self.guess)
            self.r = (self.matrix@self.guess)-self.b
            self.infinity_norm = np.around(np.linalg.norm(self.r.flatten(), np.inf),4)
            if(self.infinity_norm<self.epsilon):
                print(f"Converged after  {i} iterations")
                break
        print(self.guess)
        self.jm_guess = self.guess

# A = np.array([[4,1,1,0,1],[-1,-3,1,1,0],[2,1,5,-1,-1],[-1,-1,-1,4,0],[0,2,-1,1,4]])
# b = np.array([[6,6,6,6,6]]).T
# to_solve = matrix(A,b)
# to_solve.jacobi_method()

A_2 = np.array([[2,-1,1],[2,2,2],[-1,-1,2]])
b_2 = np.array([[-1,4,-5]]).T 

to_solve_2 = matrix(A_2,b_2)
to_solve_2.jacobi_method()
