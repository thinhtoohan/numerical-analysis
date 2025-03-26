import numpy as np
from numpy.linalg import inv,norm
np.set_printoptions(precision=4, suppress=True)
class hilbert_matrix:
    def __init__(self,n):
        self.n = n
        self.hilb_matrix = np.zeros((n, n))
        #self.true_x = np.ones(n).reshape(n, 1)
        for i in range(0, n):
            for j in range(0, n):
                self.hilb_matrix[i][j] = 1 / ((i + 1) + (j + 1) - 1)
        print(f"Hilbert Matrix with n={self.n}")
        print(self.hilb_matrix)
        
        self.inv_hilb_matrix = inv(self.hilb_matrix)
        print(f"Inverse Hilbert Matrix with n={self.n}")
        print(self.inv_hilb_matrix)
        self.hilb_matrix_norm = norm(self.hilb_matrix,ord=np.inf)
        self.inv_hilb_matrix_norm = norm(self.inv_hilb_matrix,ord=np.inf)
        print(f"Infinity norm of Hilbert Matrix with n={self.n}")
        print(self.hilb_matrix_norm)
        print(f"Infinity norm of Hilbert Matrix with n={self.n}")
        print(self.inv_hilb_matrix_norm)
        print(f"Keppa of Hilbert Matrix with n={self.n}")
        print(np.round(self.hilb_matrix_norm*self.inv_hilb_matrix_norm,4))
        
    def gaussian_elimination(self,b:np.array):
        self.augmented_matrix = np.append(self.hilb_matrix, b, axis=1)
        print(self.augmented_matrix)
        for i in range(0, self.n-1):
            pivot = self.augmented_matrix[i, i]
            for j in range(i + 1, self.n):
                multiplier = np.around(self.augmented_matrix[j, i] / pivot,4)
                self.augmented_matrix[j,i]=0 # items under pivots are zero without any calculation needed
                self.augmented_matrix[j,i+1:] = np.around((-multiplier) * (self.augmented_matrix[i,i+1:]) + self.augmented_matrix[j,i+1:],4)
        print(self.augmented_matrix)
    def back_substitution(self):
        self.x = np.zeros(self.n)
        for i in range(self.n - 1, -1, -1):
            self.x[i] = self.augmented_matrix[i, self.n]
            for j in range(i + 1, self.n):
                self.x[i] -= self.augmented_matrix[i, j] * self.x[j]
            self.x[i] = self.x[i] / self.augmented_matrix[i, i]
        print("Result:\n",self.x)

h_1 = hilbert_matrix(4)
h_2 = hilbert_matrix(5)
h_1.gaussian_elimination(np.array([[1,0,0,1]]).T)
h_1.back_substitution()