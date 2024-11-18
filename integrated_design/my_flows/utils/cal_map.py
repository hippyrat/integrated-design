import numpy as np
from collections import deque

class CalMap:
    def __init__(self):
        self.begin_node_list = []  # 起始节点列表
        self.end_node_list = []    # 终止节点列表
        self.data_node_list = []   # 所有数据节点
        self.para_node_list = []   # 所有参数节点

    def add_begin_node(self, begin_node):
        self.begin_node_list.append(begin_node)

    def add_end_node(self, end_node):
        self.end_node_list.append(end_node)

    def add_data_node(self, data_node):
        self.data_node_list.append(data_node)

    def add_para_node(self, para_node):
        self.para_node_list.append(para_node)

    def forward(self):
        if not self.begin_node_list or not self.end_node_list:
            raise ValueError("`begin_node_list` and `end_node_list` cannot be empty!")

        q_data = deque(self.begin_node_list)
        q_cal = deque()

        while q_data or q_cal:
            if q_data:
                node = q_data.popleft()
                for cal_node in node.back_node:
                    cal_node.if_forward.add(node)
                    if len(cal_node.if_forward) == len(cal_node.pre_nodes):
                        q_cal.append(cal_node)
            else:
                cal_node = q_cal.popleft()
                cal_node.forward()
                q_data.append(cal_node.back_node)

    def backward(self):
        if not self.begin_node_list or not self.end_node_list:
            raise ValueError("`begin_node_list` and `end_node_list` cannot be empty!")

        q_data = deque(self.end_node_list)
        q_cal = deque()

        while q_data or q_cal:
            if q_data:
                node = q_data.popleft()
                node.backward()
                if node.pre_node:
                    q_cal.append(node.pre_node)
            else:
                cal_node = q_cal.popleft()
                cal_node.backward()
                for data_node in cal_node.pre_nodes:
                    data_node.if_backward.add(cal_node)
                    if len(data_node.if_backward) == len(data_node.back_node):
                        q_data.append(data_node)

    def set_grad_zero(self):
        for node in self.data_node_list:
            node.grad = np.zeros_like(node.data)
            if node.pre_node:
                node.pre_node.zero_grad()

    def update_para(self, eta):
        for node in self.para_node_list:
            node.data -= eta * node.grad
            if node.pre_node:
                node.pre_node.update_para(eta)
