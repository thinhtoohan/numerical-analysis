import numpy as np
from numpy.linalg import inv
np.set_printoptions(precision=4, suppress=True)
class hilbert_matrix:
    def __init__(self,n):
        self.n = n
        self.hilb_matrix = np.zeros((n, n))
        self.true_x = np.ones(n).reshape(n, 1)
        self.epsilon = 1e-3
        for i in range(0, n):
            for j in range(0, n):
                self.hilb_matrix[i][j] = 1 / ((i + 1) + (j + 1) - 1)
        self.b = self.hilb_matrix@self.true_x
    def conjugate_gradient_method(self):
        self.guess = np.zeros(self.n).reshape(self.n, 1)
        self.r = (self.hilb_matrix@self.guess)-self.b
        self.v=-1*(self.r)
        self.t=-1*(self.r.T@self.v)/(self.v.T@self.hilb_matrix@self.v)
        for i in range(0,75):
            self.r = (self.hilb_matrix@self.guess)-self.b
            self.v=-1*(self.r)
            self.t=-1*(self.r.T@self.v)/(self.v.T@self.hilb_matrix@self.v)
            self.guess=self.guess+(self.t*self.v)
            self.r = self.hilb_matrix@self.guess-self.b
            self.infinity_norm = np.around(np.linalg.norm(self.r.flatten(), np.inf),4)
            if(self.infinity_norm<self.epsilon):
                print(f"Converged after  {i} iterations")
                break
        print(self.guess.T)
        self.cgm_guess = self.guess
    def jacobi_method(self):
        self.L = np.zeros((self.n, self.n))
        self.U = np.zeros((self.n, self.n))
        self.D = np.zeros((self.n, self.n))
        di = np.diag_indices(self.n)
        self.D[di] = self.hilb_matrix[di]
        for i in range(1, self.n):  # Start from the second row (index 1)
            for j in range(i):  # Iterate up to the diagonal element (exclusive)
                self.L[i][j] = self.hilb_matrix[i][j]
        self.L = -1*self.L
        for i in range(self.n - 1):  # Iterate up to the second-to-last row
            for j in range(i + 1, self.n):  # Iterate from the element after the diagonal
                self.U[i][j] = self.hilb_matrix[i][j]
        self.U = -1*self.U
        self.guess = np.zeros(self.n).reshape(self.n, 1)
        self.T = inv(self.D)@(self.L+self.U)
        eigvals = np.linalg.eigvals(self.T)
        spectral_radius = max(abs(eigvals))
        print(f"Spectral Radius of T: {spectral_radius:.4f}")
        self.infinity_norm_T = np.around(np.linalg.norm(self.T, np.inf),4)
        print(f"Infinity Norm of T: {self.infinity_norm_T}")
        self.c = inv(self.D)@self.b
        for i in range(0,75):
            self.guess = (self.T@self.guess)+self.c
            self.r = (self.hilb_matrix@self.guess)-self.b
            self.infinity_norm = np.around(np.linalg.norm(self.r.flatten(), np.inf),4)
            if(self.infinity_norm<self.epsilon):
                print(f"Converged after  {i} iterations")
                break
        if(self.infinity_norm>self.epsilon):
            print("Solution did not converge after 75 iterations")
            print("Final solution is : ")
        print(self.guess.T)
        self.jm_guess = self.guess

h20=hilbert_matrix(20)
print("***Conjugate Gradient Method***")
h20.conjugate_gradient_method()
print("***Jacobi's Method***")
h20.jacobi_method()





