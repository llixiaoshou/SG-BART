import numpy as np

# 原始矩阵
matrix_undir = np.array([
    ['SELF', 'case', 'case', '', '', ''],
    ['case', 'SELF', '', '', '', ''],
    ['case', '', 'SELF', '', '', ''],
    ['', '', '', 'SELF', '', ''],
    ['', '', '', '', 'SELF', ''],
    ['', '', '', '', '', 'SELF']
])

# 获取原始矩阵的行数和列数
rows, cols = matrix_undir.shape

# 创建新的矩阵，比原始矩阵多两行和两列
new_matrix = np.empty((rows + 2, cols + 2), dtype=matrix_undir.dtype)

# 将原始矩阵复制到新矩阵中
new_matrix[:-2, :-2] = matrix_undir

# 在新矩阵的最后两行增加两行数据
new_matrix[-2:, :] = ''

# 在新矩阵的最后两列增加两列数据
new_matrix[:, -2:] = ''

# 在新矩阵的右下角设置新的元素
# new_matrix[-2:, -2:] = 'SELF'

# 打印更新后的矩阵
print(new_matrix)