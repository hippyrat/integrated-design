import numpy as np
from collections import deque

class GradEngine:
    """
    反向传播引擎，用于管理和执行整个计算图的梯度计算
    """
    @staticmethod
    def backward(tensor, grad=None):
        """
        从给定的张量开始执行反向传播，计算所有依赖张量的梯度
        grad (np.ndarray, optional): 起始张量的梯度（默认为全 1）
        """
        if grad is None:
            grad = np.ones_like(tensor.data)  # 默认初始化为全 1

        # 初始化梯度图
        grad_map = {tensor: grad}

        # 获取计算图的拓扑排序
        topo_order = GradEngine._topological_sort(tensor)

        # 反向传播过程
        while topo_order:
            current_tensor = topo_order.pop()
            current_grad = grad_map.get(current_tensor, 0)  # 当前张量的累计梯度

            # 如果有生成此张量的计算节点
            if current_tensor._grad_fn is not None:
                # 计算每个父节点的梯度贡献
                parent_grads = current_tensor._grad_fn.parent_grads(current_grad)

                for parent, grad_contrib in zip(current_tensor._grad_fn.parents, parent_grads):
                    if parent.requires_grad:
                        # 累积梯度
                        if parent not in grad_map:
                            grad_map[parent] = grad_contrib
                        else:
                            grad_map[parent] += grad_contrib

            # 更新当前张量的梯度
            if current_tensor.requires_grad:
                current_tensor.grad = current_grad

    @staticmethod
    def _topological_sort(tensor):
        """
        对计算图进行拓扑排序，返回节点的执行顺序
        返回一个list

        """
        visited = set()
        order = []

        def visit(node):
            if node not in visited:
                visited.add(node)
                if node._grad_fn is not None:
                    for parent in node._grad_fn.parents:
                        visit(parent)
                order.append(node)

        visit(tensor)
        return order
