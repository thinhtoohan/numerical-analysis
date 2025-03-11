import numpy as np
from numpy.linalg import inv
class matrix:
    def __init__(self,matrix:np.array,b:np.array):
        self.matrix = matrix
        self.b = b
        self.n = matrix.shape[0]
        self.epsilon = 1e-3
    def sor_method(self,w:float):
        self.w = w
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
        self.T_w = inv(self.D-(self.w*self.L))@(((1-self.w)*self.D)+(self.w*self.U))
        self.c_w = self.w*inv(self.D-(w*self.L))@self.b
        for i in range(0,50):
            self.guess = (self.T_w@self.guess)+self.c_w
            print(self.guess)
            self.r = (self.matrix@self.guess)-self.b
            self.infinity_norm = np.around(np.linalg.norm(self.r.flatten(), np.inf),4)
            if(self.infinity_norm<self.epsilon):
                print(f"Converged after  {i} iterations")
                
                break
        print(self.guess)

A_1 = np.array([[4,1,1,0,1],[-1,-3,1,1,0],[2,1,5,-1,-1],[-1,-1,-1,4,0],[0,2,-1,1,4]])
b_1 = np.array([[6,6,6,6,6]]).T

to_solve = matrix(A_1,b_1)
to_solve.sor_method(1.2)