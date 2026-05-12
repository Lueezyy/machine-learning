# Machine Learning Library

Input features and weights:
```math
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_m \end{bmatrix},
\enspace \mathbf{w} = \begin{bmatrix} w_1 \\ w_2 \\ \vdots \\ w_m \end{bmatrix}
```
Net input function:  
```math
z = \mathbf{w}^T \mathbf{x} + b
```
### 1. [Perceptron](perceptron.py)  
Threshold function:  
```math
\sigma(z) = \left\{ \begin{matrix} 1 & z \ge 0 \\ 0 & z < 0 \end{matrix} \right.  
```
Predicted class label:  
```math
\hat{y} = \sigma(z)
```
Error:  
```math
e = (y^{(i)} - \hat{y}^{(i)})
```
Weight and bias updates:  
```math
\Delta \mathbf{w} = \eta \cdot e \cdot \mathbf{x}^{(i)}, \quad \Delta b = \eta \cdot e
```
```math
\mathbf{w} := \mathbf{w} + \Delta \mathbf{w}, \quad b := b + \Delta b
```

### 2. [ADAptive LInear NEuron (Adaline)](adaline.py)