import numpy as np

class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=None):
        """
        初始化卷积层。

        参数：
        - in_channels: 输入的通道数（如 RGB 图像的通道数为 3）。
        - out_channels: 输出的通道数，即卷积核的数量。
        - kernel_size: 卷积核的大小（假定为正方形，如 3 表示 3x3）。
        - stride: 卷积核滑动的步幅，默认值为 1。
        - padding: 输入的边缘填充大小，默认值为 0。
        - activation: 激活函数（如 np.tanh 或其他函数），默认值为 None。
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation

        # 初始化卷积核权重
        # 权重的形状为 (out_channels, in_channels, kernel_size, kernel_size)
        # 使用小的随机值初始化，避免一开始权重为零导致的梯度问题
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01

        # 初始化偏置为零
        # 每个输出通道对应一个偏置
        self.bias = np.zeros((out_channels, 1))

    def forward(self, input_data):
        """
        执行前向传播，计算卷积操作的输出。

        参数：
        - input_tensor: 输入的数据，形状为 (batch_size, in_channels, height, width)。

        返回：
        - 输出的特征图，形状为 (batch_size, out_channels, output_height, output_width)。
        """
        batch_size, _, input_height, input_width = input_data.shape

        # 计算输出特征图的高度和宽度
        # 输出尺寸公式：output_size = (input_size - kernel_size + 2 * padding) // stride + 1
        output_height = (input_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        output_width = (input_width - self.kernel_size + 2 * self.padding) // self.stride + 1

        # 初始化输出特征图的张量
        feature_map = np.zeros((batch_size, self.out_channels, output_height, output_width))

        # 如果需要填充，在输入的高和宽两侧添加零
        if self.padding > 0:
            input_data = np.pad(
                input_data,
                pad_width=((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )

        # 遍历每个样本（batch）、每个输出通道，以及每个滑动窗口的位置
        for b in range(batch_size):  # 遍历 batch 中的每个样本
            for c_out in range(self.out_channels):  # 遍历每个输出通道
                for i in range(output_height):  # 遍历输出特征图的高度
                    for j in range(output_width):  # 遍历输出特征图的宽度
                        # 提取滑动窗口的区域，窗口大小等于卷积核大小
                        region = input_data[
                            b,  # 当前样本
                            :,  # 所有输入通道
                            i * self.stride:i * self.stride + self.kernel_size,  # 滑动窗口的行范围
                            j * self.stride:j * self.stride + self.kernel_size  # 滑动窗口的列范围
                                 ]
                        # 计算卷积操作：局部区域与卷积核对应元素的点积，再加上偏置
                        feature_map[b, c_out, i, j] = np.sum(region * self.weights[c_out]) + self.bias[c_out].item()

        # 如果设置了激活函数，应用激活函数
        if self.activation:
            feature_map = self.activation(feature_map)

        return feature_map


# 示例使用
# 生成随机输入数据，形状为 (batch_size=1, in_channels=3, height=32, width=32)
input_tensor = np.random.randn(1, 3, 32, 32)

# 初始化卷积层
# 输入通道数为 3，输出通道数为 16，卷积核大小为 3x3，步幅为 1，填充为 1，激活函数为 tanh
conv = ConvolutionalLayer(3, 16, 3, stride=1, padding=1, activation=np.tanh)

# 执行前向传播，得到输出特征图
output_tensor = conv.forward(input_tensor)

# 打印输出特征图的形状
# 期望输出形状为 (batch_size=1, out_channels=16, height=32, width=32)
print(output_tensor.shape)
