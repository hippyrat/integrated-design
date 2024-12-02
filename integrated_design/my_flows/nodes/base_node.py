from integrated_design.my_flows.tensor import Tensor

class CalNode:
    """
    计算节点的基类，定义基本的接口
    """
    def __init__(self, *parents):
        """
        初始化计算节点，记录父节点
        """
        self.parents = parents  # 父节点列表

    def forward(self):
        """
        执行前向传播，返回结果张量
        """
        raise NotImplementedError("Forward propagation must be implemented in subclasses.")

    def parent_grads(self, grad_output):
        """
        计算传递到每个父节点的梯度
        """
        raise NotImplementedError("Parent gradients must be implemented in subclasses.")

    def backward(self, grad_output):
        """
        执行反向传播，计算父节点的梯度
        """
        for parent, grad in zip(self.parents, self.parent_grads(grad_output)):
            if parent.requires_grad:
                parent.grad += grad
