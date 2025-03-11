import numpy as np
from numpy.linalg import inv
np.set_printoptions(precision=4, suppress=True)
class matrix:
    def __init__(self,matrix:np.array,true_x:np.array,b:np.array):
        self.matrix = matrix
        self.n = matrix.shape[0]
        self.true_x = true_x
        self.epsilon = 1e-6
        self.b = b
    def conjugate_gradient_method(self):
        self.guess = np.zeros(self.n).reshape(self.n, 1)
        print(self.guess)
        print(self.matrix@self.guess)
        self.r = (self.matrix@self.guess)-self.b
        print(self.r)
        self.guess = np.zeros(self.n).reshape(self.n, 1)
        print(self.guess)
        self.v=-1*(self.r)
        print(self.v)
        self.t=-1*(self.r.T@self.v)/(self.v.T@self.matrix@self.v)
        print(self.t)
        for i in range(0,1500):
            self.r = (self.matrix@self.guess)-self.b
            self.v=-1*(self.r)
            self.t=-1*(self.r.T@self.v)/(self.v.T@self.matrix@self.v)
            self.guess=self.guess+(self.t*self.v)
            self.r = self.matrix@self.guess-self.b
            self.infinity_norm = np.around(np.linalg.norm(self.r.flatten(), np.inf),4)
            if(self.infinity_norm<self.epsilon):
                print(f"Converged after  {i} iterations")
                break
        print(self.guess)
        self.cgm_guess = self.guess

m = np.array([[1, 0.5, 0.333],[0.5, 0.333, 0.25],[0.333, 0.25, 0.2]])
tru_x = np.array([[1,-1,1]])
tru_b = np.array([[0.833,0.417,0.283]]).T

to_solve = matrix(m,tru_x,tru_b)
to_solve.conjugate_gradient_method()