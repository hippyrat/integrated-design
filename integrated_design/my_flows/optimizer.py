import numpy as np

# def get_loss(x, y):


# def get_gradient(x, y):


class Adam:
    def __init__(self, params, lr = 0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas # 动量估计的衰减率
        self.eps = eps # 防止除以0的小值

        # 初始化
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0 # 时间步计数器

    def step(self, grads):
        self.t += 1  # Update time step

        for i in range(len(self.params)):
            # Get the gradients of the current parameter
            grad = grads[i]

            # 更新m
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # 更新v
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            #
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)


            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # 更新参数
            self.params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    # def zero_grad(self):
