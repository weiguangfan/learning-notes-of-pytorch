"""张量初始化、张量属性、张量运算、tensor 与 numpy 的转化"""
import numpy as np
import torch

# 第一部分：张量初始化
# data = [[1, 2], [3, 4]]
# print(type(data))

# 方式一：直接生成张量
# x_data = torch.tensor(data)
# print(type(x_data))
# print(x_data)

# 方式二：通过numpy数组生成张量
# np_array = np.array(data)
# print(type(np_array))
# print(np_array)

# 方式三：通过已有的张量生成新的张量
# x_ones = torch.ones_like(x_data)  # 保留x_data结构和属性
# print(type(x_ones))
# print(f"ones tensor: \n {x_ones} \n ")
#
# x_rand = torch.rand_like(x_data, dtype=torch.float)  # 重写x_data的数据类型 int -> float
# print(type(x_rand))
# print(f"random tensor: \n {x_rand} \n ")

# 方式四：通过指定数据维度来生成张量
# shape = (2, 3,)
# rand_tensor = torch.rand(shape)
# print(f"random tensor: \n {rand_tensor} \n ")
# ones_tensor = torch.ones(shape)
# print(f"ones tensor: \n {ones_tensor} \n ")
# zeros_tensor = torch.zeros(shape)
# print(f"zeros tensor: \n {zeros_tensor} \n ")

# 第二部分：张量属性
# tensor = torch.rand(3, 4)
# print(type(tensor))
# print(tensor)
# 维数
# print(f"shape of tensor: \n {tensor.shape} \n")
# 数据类型
# print(f"datatype of tensor: \n {tensor.dtype} \n")
# 数据存储的设备
# print(f"device of tensor: \n {tensor.device} \n")

# 第三部分：张量运算
# 3.1 张量的索引和切片
# tensor = torch.ones(4, 4)
# print(type(tensor))
# print(tensor)
# tensor[:, 1] = 0  # 第一列元素全部赋值为0
# print(tensor)

# 3.2 张量的拼接
# t1 = torch.cat([tensor, tensor, tensor], dim=1)  # 按照行方向拼接
# print(t1)

# 3.3 张量的乘积和矩阵乘法
# 3.3.1张量的乘积
# 逐个元素对应相乘再相加
# print(f"tensor.mul(tensor): \n {tensor.mul(tensor)} \n")
# 等价于以下
# print(f"tensor * tensor: \n {tensor * tensor} \n")

# 3.3.2矩阵的乘积
# 矩阵A右乘矩阵B
# print(tensor.__rmatmul__(tensor.T))
# print(tensor.T @ tensor)
# 矩阵A左乘矩阵B
# print(tensor @ tensor.T)

# 3.4 自动赋值运算
# 方法后面有_作为后缀
# 所有元素均加5
# print(tensor.add_(5))

# 第四部分：tensor 与 numpy 的转化
# 两者公用一块内存区域，一个改变，另一个也改变
# 4.1 由张量转化为numpy.array
# 维度(1,5)
# t = torch.ones(5)
# print(type(t))
# print(f"t:\n {t} \n")
# t.numpy()
# n = t.numpy()
# print(type(n))
# print(f"n:\n {n} \n")
# 改变一个的值，另一个也改变
# t.add_(1)
# print(f"t:\n {t} \n")
# print(f"n:\n {n} \n")

# 4.2 由numpy.array转为张量
n = np.ones(5)
print(type(n))
print(f"shape of n:\n {n.shape} \n")
print(f"n:\n {n} \n")
# torch.from_numpy()
t = torch.from_numpy(n)
print(f"n: \n {type(t)} \n ")
print(f"n: \n {t.shape} \n ")
print(f"n: \n {t} \n ")
# 改变一个的值，另一个也改变
np.add(n, 1, out=n)
print(f"n: \n {n} \n ")
print(f"n: \n {t} \n ")


