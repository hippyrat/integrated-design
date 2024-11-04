class Context:
    """
    在前向传播中保存反向传播所需的中间变量。
    """
    def __init__(self):
        self._saved_tensors = []

    def save_for_backward(self, *tensors):
        self._saved_tensors.extend(tensors)

    @property
    def saved_tensors(self):
        return self._saved_tensors
