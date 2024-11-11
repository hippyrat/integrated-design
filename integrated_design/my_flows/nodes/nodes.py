import numpy as np

class DataNode:
    def __init__(self, data, requires_grad=True):
        # 初始化数据节点
        self.data = np.array(data, dtype=np.float32)    # 存储数值
        self.requires_grad = requires_grad  # 是否需要计算梯度
        self.grad = np.zeros_like(self.data)    # 初始化梯度为0
        self.pre_node = None    # 前驱节点
        self.back_node = [] # 后继节点
        self.if_backward = set()    # 标记是否进行了反向传播

    def set_pre_node(self, pre_node):
        self.pre_node = pre_node

    def add_back_node(self, back_node):
        self.back_node.append(back_node)

    def set_data(self, data):
        self.data = np.array(data, dtype=np.float32)

    def set_grad(self, grad):
        self.grad = np.array(grad, dtype=np.float32)

    def backward(self):
        # 初始化反向传播，清空反向传播记录
        if not self.requires_grad:
            return
        self.if_backward.clear()
        self.grad = np.ones_like(self.data)
        if self.pre_node:
            self.pre_node.backward(self.grad)

# 常量节点
class ConstNode(DataNode):
    def __init__(self, data):
        super().__init__(data, requires_grad=False)

# 计算节点
class CalNode:
    def __init__(self):
        self.pre_nodes = [] # 输入节点
        self.back_node = None  # 输出节点
        self.if_forward = set() # 用于判断是否进行前向传播

    def set_pre_nodes(self, pre_nodes):
        self.pre_nodes = pre_nodes

    def set_back_node(self, back_node):
        self.back_node = back_node

    def add_pre_node(self, pre_node):
        self.pre_nodes.append(pre_node)

    def forward(self):
        # 前向传播的具体实现会由子类覆盖
        raise NotImplementedError

    def backward(self, grad_output):
        # 反向传播的具体实现会由子类覆盖,如果子类里直接调用而没有具体实现，就会报错
        raise NotImplementedError

    def zero_grad(self):
        for node in self.pre_nodes:
            node.grad = np.zeros_like(node.grad)

# 加法节点
class AddNode(CalNode):
    def forward(self, x, y):
        self.x = x
        self.y = y
        self.back_node = DataNode(x.data + y.data, requires_grad=(x.requires_grad or y.requires_grad))
        self.back_node.set_pre_node(self)
        x.add_back_node(self)
        y.add_back_node(self)
        return self.back_node

    def backward(self, grad_output):
        if self.x.requires_grad:
            self.x.grad += grad_output
        if self.y.requires_grad:
            self.y.grad += grad_output

# 减法节点
class SubNode(CalNode):
    def forward(self, x, y):
        self.x = x
        self.y = y
        self.back_node = DataNode(x.data - y.data, requires_grad=(x.requires_grad or y.requires_grad))
        self.back_node.set_pre_node(self)
        x.add_back_node(self)
        y.add_back_node(self)
        return self.back_node

    def backward(self, grad_output):
        if self.x.requires_grad:
            self.x.grad += grad_output
        if self.y.requires_grad:
            self.y.grad -= grad_output

# 惩罚节点
class MulNode(CalNode):
    def forward(self, x, y):
        self.x = x
        self.y = y
        self.back_node = DataNode(x.data * y.data, requires_grad=(x.requires_grad or y.requires_grad))
        self.back_node.set_pre_node(self)
        x.add_back_node(self)
        y.add_back_node(self)
        return self.back_node

    def backward(self, grad_output):
        if self.x.requires_grad:
            self.x.grad += grad_output * self.y.data
        if self.y.requires_grad:
            self.y.grad += grad_output * self.x.data

# 除法节点
class DivNode(CalNode):
    def forward(self, x, y):
        self.x = x
        self.y = y
        self.back_node = DataNode(x.data / y.data, requires_grad=(x.requires_grad or y.requires_grad))
        self.back_node.set_pre_node(self)
        x.add_back_node(self)
        y.add_back_node(self)
        return self.back_node

    def backward(self, grad_output):
        if self.x.requires_grad:
            self.x.grad += grad_output / self.y.data
        if self.y.requires_grad:
            self.y.grad -= grad_output * self.x.data / (self.y.data ** 2)
"""
使用示例（GPT生成的）
# 初始化数据节点
x = DataNode([2.0], requires_grad=True)
y = DataNode([3.0], requires_grad=True)

# 构建加法节点
add_op = AddNode()
z = add_op.forward(x, y)  # z = x + y

# 前向传播
print("z data:", z.data)  # 应输出 z = 5.0

# 反向传播
z.backward()
print("dz/dx:", x.grad)  # 应输出 dz/dx = 1.0
print("dz/dy:", y.grad)  # 应输出 dz/dy = 1.0

"""
