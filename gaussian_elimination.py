import numpy as np
np.set_printoptions(precision=4, suppress=True)
class hilbert_matrix:
    def __init__(self,n):
        self.n = n
        self.hilb_matrix = np.zeros((n, n))
        self.true_x = np.ones(n).reshape(n, 1)
        self.multiplication_count = 0
        for i in range(0, n):
            for j in range(0, n):
                self.hilb_matrix[i][j] = 1 / ((i + 1) + (j + 1) - 1)
        self.augmented_matrix = np.append(self.hilb_matrix, self.hilb_matrix @ self.true_x, axis=1)
        print(self.augmented_matrix)
    def gaussian_elimination(self):
        for i in range(0, self.n-1):
            pivot = self.augmented_matrix[i, i]
            for j in range(i + 1, self.n):
                multiplier = self.augmented_matrix[j, i] / pivot
                self.augmented_matrix[j,i]=0 # items under pivots are zero without any calculation needed
                self.multiplication_count += 1
                self.augmented_matrix[j,i+1:] = (-multiplier) * (self.augmented_matrix[i,i+1:]) + self.augmented_matrix[j,i+1:]
                self.multiplication_count += self.n-i
        '''
        print(self.augmented_matrix)
        '''
    def back_substitution(self):
        self.x = np.zeros(self.n)
        for i in range(self.n - 1, -1, -1):
            self.x[i] = self.augmented_matrix[i, self.n]
            for j in range(i + 1, self.n):
                self.x[i] -= self.augmented_matrix[i, j] * self.x[j]
                self.multiplication_count += 1
            self.x[i] = self.x[i] / self.augmented_matrix[i, i]
            self.multiplication_count+=1
        print("Result:\n",self.x)
    def calculate_error(self):
        self.error = self.true_x.T-self.x
        print("Error:\n",self.error.flatten())
    def calculate_norms(self):
        self.infinity_norm = np.linalg.norm(self.error.flatten(), np.inf)
        self.l2_norm = np.around(np.linalg.norm(self.error),4)  
        print("Infinity Norm:\n",self.infinity_norm)
        print("L2 Norm:\n",self.l2_norm)
    def print_multiplication_count(self):
        print(f"Multiplication Count for n={self.n}:\n",self.multiplication_count)


h_3 = hilbert_matrix(12)
h_3.gaussian_elimination()
h_3.back_substitution()
h_3.calculate_error()
h_3.calculate_norms()
h_3.print_multiplication_count()


# print("Hilbert Matrix with n=12")
# h_12 = hilbert_matrix(12)
# h_12.gaussian_elimination()
# h_12.back_substitution()
# h_12.calculate_error()
# h_12.calculate_norms()
# h_12.print_multiplication_count()

# print("Hilbert Matrix with n=13")
# h_13 = hilbert_matrix(13)
# h_13.gaussian_elimination()
# h_13.back_substitution()
# h_13.calculate_error()
# h_13.calculate_norms()
# h_13.print_multiplication_count()