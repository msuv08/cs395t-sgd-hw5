# SGD for Linear Least Squares (HW5, Q4)

## Overview

This repository contains the Python implementation of Stochastic Gradient Descent (SGD) for solving the linear least squares problem, which is formulated as minimizing the loss function `f(w) = 1/2 ||Aw - b||^2`. This code is for HW5, Q4 and demonstrates how SGD converges over iterations for different variances of noise in the observations.

## Implementation Details

The `stochastic_gradient_descent` function in `sgd.py` performs the core SGD optimization, updating parameters `w` iteratively based on the gradient of the loss function.

Two types of experiments were conducted:
1. **Standard Matrix A**: `A` is a `1000x1000` matrix with i.i.d. Gaussian entries of variance `1/d`.
2. **Large Matrix A**: `A` is a `10000x1000` matrix where each `j`-th row has i.i.d. Gaussian entries with variance scaled by `1/sqrt(1000*(j))`.

The target vector `b` is generated as `A * 1 + ε` where `1` is the all-ones vector and `ε` is noise with Gaussian entries of varying variances: 1, 0.1, 0.01, and 0.

## Results

Results are presented in two sets of convergence plots:
1. **Convergence with Standard Matrix A**:
   ![Convergence with Standard Matrix A](/figs/sgd-A.png)

2. **Convergence with Large Matrix A**:
   ![Convergence with Large Matrix A](/figs/sgd-large-A.png)

These plots above illustrate the expected decreasing trend in loss as the number of iterations increases, showcasing the effect of the noise variance and the matrix condition on the SGD convergence.