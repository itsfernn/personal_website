---
title: "Gradient Descent"
date: 2025-05-31T20:56:40+09:00
author: "Lukas Hofbauer"
github: "https://github.com/itsfernn/ml-daily-challenge/blob/main/gradient-descent/gradient-descent.ipynb"
cover:
  image: "cover.png"
---

Gradient descent is a general-purpose optimization algorithm that lies at the heart of many machine learning applications. The idea is to iteratively adjust a set of parameters, $\theta$, to minimize a given cost function.

Like a ball rolling downhill, gradient descent uses the local gradient of the cost function with respect to $\theta$ to guide its steps in the direction of steepest descent.


![png](output_1_0.png)


## The Role of the Learning Rate

The most critical hyperparameter in gradient descent is the **learning rate**. Choosing the right learning rate is crucial to ensure the algorithm converges efficiently.

* If the learning rate is too **small**, the algorithm will take a long time to converge.
* If it’s too **large**, the steps might overshoot the minimum, causing the algorithm to diverge or oscillate wildly.

The figure above shows how different learning rates affect convergence.



![png](output_3_0.png)


Another potential pitfall is that gradient descent can get **stuck in local minima**. For non-convex functions, this can result in suboptimal solutions.

---

## Gradient Descent for Linear Regression

Let's revisit yesterday’s linear regression example. Although linear regression has a closed-form solution, it serves as a great illustration of how gradient descent operates.

We define a cost function \$J(w, b)\$, then derive its gradient with respect to both parameters:

$$
J(w,b) := \frac{1}{2m} \sum_{i=1}^{m} (w \cdot x_i + b - y_i)^2
$$

$$
\frac{\partial J}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} (w \cdot x_i + b - y_i) \cdot x_i
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (w \cdot x_i + b - y_i)
$$


```python
import numpy as np
import matplotlib.pyplot as plt

def J(x, y, w, b):
    y_hat = w * x + b
    return (1/(2*m)) * np.sum((y_hat - y)**2)


def J_grad(x, y, w,b):
    error = w * x + b - y
    dw = np.average(error * x)
    db = np.average(error)
    return dw, db


# Generate data
np.random.seed(0)
x = np.linspace(0, 10, 500)
y = w * x + b
y += np.random.randn(*x.shape) * 3 # Add noise
```

## Variants of Gradient Descent

We'll compare three different versions of gradient descent:

1. **Batch Gradient Descent** – Uses the entire dataset to compute the gradient at each step.
2. **Stochastic Gradient Descent (SGD)** – Updates parameters using one sample at a time. Faster updates but more noise.
3. **Mini-Batch Gradient Descent** – A compromise between the two: updates using small batches of samples.

While SGD allows for faster updates per epoch, it introduces high variance. Mini-batch gradient descent reduces this noise while still benefiting from faster convergence than batch GD.


```python
def gradient_decent(x, y, init_w, init_b, alpha, num_epochs):
    w = init_w
    b = init_b
    path = [(w,b)]
    for _ in range(num_epochs):
        dw, db = J_grad(x,y,w,b)
        w -= alpha * dw
        b -= alpha * db
        path.append((w,b))
    return path

def stochastic_gradient_decent(x, y, init_w, init_b, alpha, num_epochs):
    w = init_w
    b = init_b
    path = [(w,b)]
    m = len(x)
    for _ in range(num_epochs):
        indices = np.random.permutation(m)
        new_x = x[indices]
        new_y = y[indices]
        for i in range(m):
            cx = new_x[i]
            cy = new_y[i]
            dw, db = J_grad(cx,cy,w,b)
            w -= alpha * dw
            b -= alpha * db
            path.append((w,b))
    return path

def minibatch_gradient_decent(x, y, init_w, init_b, alpha, num_epochs, batch_size):
    w = init_w
    b = init_b
    path = [(w,b)]
    m = x.shape[0]
    for _ in range(num_epochs):
        indices = np.random.permutation(m)
        new_x = x[indices]
        new_y = y[indices]
        for i in range(0, m, batch_size):
            cx = new_x[i:i+batch_size]
            cy = new_y[i:i+batch_size]
            dw, db = J_grad(cx,cy,w,b)
            w -= alpha * dw
            b -= alpha * db
            path.append((w,b))
    return path
```



![png](output_10_0.png)



Notice that the path curves as it approaches the minimum. This happens because the scales of \$w\$ and \$b\$ are different, causing the cost surface to be skewed.

---

## Feature Scaling and Faster Convergence

We can fix the skewed path by **rescaling** the input features. After standardizing the input \$x\$, the cost surface becomes more isotropic (circular contours), allowing for more direct paths to the minimum and faster convergence.


![png](output_12_0.png)



## Summary

Gradient descent is a powerful and versatile optimization tool. Understanding how the learning rate, local minima, and feature scaling affect convergence is essential for training machine learning models efficiently. The various flavors of gradient descent offer trade-offs between speed and stability, and choosing the right one depends on the problem and dataset size.
