import numpy as np

# SGD(Stochastic Gradient Descent): 확률적 경사 하강법
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    def update(self, params, grads):
        for i in range(params.shape[0]):
            params[i] -= self.lr * grads[i]
            
# Momentum
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    def update(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)
        for i in range(len(params)):
            self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
            params[i] += self.v[i]

# NAG(Nesterov Accelerated Gradient)
class NAG:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    def update(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)
        for i in range(len(params)):
            params[i] += self.momentum * self.momentum * self.v[i]
            params[i] -= (1 + self.momentum) * self.lr * grads[i]
            self.v[i] *= self.momentum
            self.v[i] -= self.lr * grads[i]

# AdaGrad(Adaptive Gradient)
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
    def update(self, params, grads):
        if self.h is None:
            self.h = np.zeros_like(params)
        for i in range(len(params)):
            self.h[i] = round(self.h[i] + grads[i] * grads[i], 4)
            params[i] = round(params[i] - self.lr * grads[i] / (np.sqrt(self.h[i]) + 1e-7), 4)

# RMSProp
class RMSProp:
    def __init__(self, lr=0.01, gamma=0.75):    # gamma: forgetting factor(decay rate)
        self.lr = lr
        self.gamma = gamma      # gamma가 클수록 과거가 중요하고, 작을수록 현재(gradient)가 중요
        self.h = None
    def update(self, params, grads):
        if self.h is None:
            self.h = np.zeros_like(params)
        for i in range(len(params)):
            self.h[i] = round(self.gamma * self.h[i] + (1 - self.gamma) * grads[i] * grads[i], 4)
            params[i] = round(params[i] - self.lr * grads[i] / (np.sqrt(self.h[i]) + 1e-7), 4)

# Adam
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr, self.beta1, self.beta2 = lr, beta1, beta2
        self.iter, self.m, self.v = 0, None, None
    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.iter += 1
        lr_t = self.lr * np.sqrt(1. - self.beta2**self.iter) / (1. - self.beta1**self.iter)
        
        for i in range(len(params)):
            self.m[i] = round(self.beta1 * self.m[i] + (1. - self.beta1) * grads[i], 4)
            # self.m[i] += (1. - self.beta1) * (grads[i] - self.m[i])
            self.v[i] = round(self.beta2 * self.v[i] + (1. - self.beta2) * grads[i]**2, 4)
            # self.v[i] += (1. - self.beta2) * (grads[i]**2 - self.v[i])
            params[i] = round(params[i] - lr_t * self.m[i] / (np.sqrt(self.v[i] + 1e-7)), 4)