import numpy as np
from integrated_design.my_flows.auto_grad.grad_engine import GradEngine

class Tensor:
    """
    张量类，是数据容器，也是动态图的一部分
    """
    def __init__(self, data, requires_grad=True):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._grad_fn = None

    def backward(self, grad=None):
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward on a tensor that does not require gradients.")
        GradEngine.backward(self, grad)

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    # 推迟运算节点的导入
    def __add__(self, other):
        from integrated_design.my_flows.nodes.arithmetic import AddNode  # 延迟导入
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return AddNode(self, other).forward()

    def __sub__(self, other):
        from integrated_design.my_flows.nodes.arithmetic import SubNode  # 延迟导入
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return SubNode(self, other).forward()

    def __mul__(self, other):
        from integrated_design.my_flows.nodes.arithmetic import MulNode  # 延迟导入
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return MulNode(self, other).forward()

    def __truediv__(self, other):
        from integrated_design.my_flows.nodes.arithmetic import DivNode  # 延迟导入
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return DivNode(self, other).forward()

    def __repr__(self):
        return (f"Tensor(data={self.data}, requires_grad={self.requires_grad}, "
                f"shape={self.data.shape})")
