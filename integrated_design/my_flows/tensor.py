import numpy as np
from function import Add  # 导入 Add 类

class Tensor:
    """
    主要的数据结构，包含前向传播和反向传播功能。
    """
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None

    def __add__(self, other):
        return Add.apply(self, other)

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        if self.grad_fn:
            self.grad_fn.backward(self.grad)
