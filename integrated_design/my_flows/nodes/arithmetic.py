from integrated_design.my_flows.tensor import Tensor
from integrated_design.my_flows.nodes.base_node import CalNode  # 确保从正确的模块导入 CalNode

class AddNode(CalNode):
    def __init__(self, *parents):
        super().__init__(*parents)

    def forward(self):
        result = Tensor(self.parents[0].data + self.parents[1].data, requires_grad=True)
        result._grad_fn = self
        return result

    def parent_grads(self, grad_output):
        return [grad_output, grad_output]


class SubNode(CalNode):
    def __init__(self, *parents):
        super().__init__(*parents)

    def forward(self):
        result = Tensor(self.parents[0].data - self.parents[1].data, requires_grad=True)
        result._grad_fn = self
        return result

    def parent_grads(self, grad_output):
        return [grad_output, -grad_output]

class MulNode(CalNode):
    def __init__(self, *parents):
        super().__init__(*parents)

    def forward(self):
        result = Tensor(self.parents[0].data * self.parents[1].data, requires_grad=True)
        result._grad_fn = self
        return result

    def parent_grads(self, grad_output):
        return [
            grad_output * self.parents[1].data,
            grad_output * self.parents[0].data,
        ]

class DivNode(CalNode):
    def __init__(self, *parents):
        super().__init__(*parents)

    def forward(self):
        result = Tensor(self.parents[0].data / self.parents[1].data, requires_grad=True)
        result._grad_fn = self
        return result

    def parent_grads(self, grad_output):
        x, y = self.parents[0].data, self.parents[1].data
        return [
            grad_output / y,
            -grad_output * x / (y ** 2),
        ]

