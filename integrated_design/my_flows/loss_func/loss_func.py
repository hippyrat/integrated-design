import numpy as np

class LossFunction:
    def __call__(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        self.loss_value = self.forward(predictions, targets)
        return self.loss_value

    def forward(self, predictions, targets):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

class CELossNode(LossFunction):
    def forward(self, predictions, targets):
        """
        前向传播：计算交叉熵损失
        """

        # 计算 softmax
        exp_predictions = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))  # 防止数值溢出
        self.probabilities = exp_predictions / np.sum(exp_predictions, axis=1, keepdims=True)

        # 交叉熵损失
        batch_size = predictions.shape[0]
        loss = -np.sum(targets * np.log(self.probabilities + 1e-10)) / batch_size
        return loss

    def backward(self):
        """
        反向传播：计算 predictions 的梯度
        """
        batch_size = self.predictions.shape[0]
        self.grad_predictions = (self.probabilities - self.predictions) / batch_size
        return self.grad_predictions

class MSELoss(LossFunction):
    def forward(self, predictions, targets):
        """
        前向传播：计算均方误差损失
        """

        loss = np.mean((predictions - targets) ** 2)
        return loss

    def backward(self):
        """
        反向传播：计算预测值的梯度
        """
        batch_size = self.predictions.shape[0]
        grad_predictions = 2 * (self.predictions - self.targets) / batch_size
        return grad_predictions

"""
# 示例输入
y_pred = np.array([[2.0, 3.0], [4.0, 5.0]])  # 预测值
y_true = np.array([[1.0, 2.0], [3.0, 4.0]])  # 真实值

# 创建 MSE 损失实例
loss_function = MSELoss()

# 前向传播：计算损失值
loss = loss_function(y_pred, y_true)
print("Loss:", loss)

# 反向传播：计算梯度
grad = loss_function.backward()
print("Gradients (predictions):")
print(grad)
"""