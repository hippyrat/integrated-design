import numpy as np
"""
损失函数模块，包含：MSELoss
"""
class MSELoss:
    def __call__(self, y_pred, y_true):
        diff = y_pred - y_true
        return (diff * diff).mean()
