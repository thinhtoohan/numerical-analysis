# Matrix Methods and Hilbert Matrix Solver

This repository contains implementations of various matrix methods, with a particular focus on efficiently solving the notoriously ill-conditioned Hilbert matrix.

## Implemented Methods

The following numerical linear algebra methods are currently implemented:

* **Gaussian Elimination:** A fundamental direct method for solving systems of linear equations.
* **Gauss-Seidel Method:** An iterative method for solving linear systems.
* **Jacobi Method:** Another iterative method for solving linear systems, similar to Gauss-Seidel but with simultaneous updates.
* **Conjugate Gradient Method:** An efficient iterative method particularly well-suited for symmetric positive-definite systems, such as those arising from the Hilbert matrix.

## Hilbert Matrix Optimization

A significant portion of this project focuses on optimizing the implemented methods for solving the Hilbert matrix. The Hilbert matrix is a classic example of a matrix with a high condition number, making it challenging for many numerical methods. The optimizations aim to improve:

* **Accuracy:** Minimizing the propagation of rounding errors.
* **Efficiency:** Reducing the computational cost and execution time.
* **Stability:** Enhancing the robustness of the methods when dealing with ill-conditioned systems.

## Usage

To use these implementations, you will need a Python environment with the NumPy library installed.

```bash
pip install numpy
