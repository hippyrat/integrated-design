# function.py
from .context import Context
import numpy as np
from .tensor import Tensor  # 使用相对导入

class Function:
    @staticmethod
    def apply(*inputs):
        ctx = Context()
        result = ctx.forward(*inputs)
        result.grad_fn = ctx
        return result

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *grad_outputs):
        raise NotImplementedError


class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(a.data + b.data, requires_grad=(a.requires_grad or b.requires_grad))

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_output * np.ones_like(a.data)
        grad_b = grad_output * np.ones_like(b.data)
        return grad_a, grad_b
