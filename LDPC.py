'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Author: 袁瑞隆
Company:西安交通大学
Date:2021.03.28
Environment:Python3.9
IDE：Pycharm2020.3.5
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np
import random
import math

'''第一步生成bit流'''
# 输入生成比特流长度
A = int(input('请输入bit流长度'))

# 创建一个长度为A的全零数组
bit = np.zeros(A, dtype=int)

# 生成随机二进制数并输入数组bit
for i in range(0, A):
    a = random.randint(0, 1)
    bit[i] = a

'''第二步添加TBCRC'''
# 创建用于CRC编码的生成多项式g_CRC24A(D)
g_24A = [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1]

# 生成多项式的长度L为24
L = 24

# 向bit流后补L位0，为模2除法做准备
zero_L = np.zeros(L, dtype=int)
bit = np.append(bit, zero_L)

# 二进制除法得到余数
# 定义中间变量mid与每一位计算结果ans
mid = np.zeros((L + 1), dtype=int)
ans = 0

# 定义余数
p = np.zeros(L, dtype=int)
for i in range(0, L + 1):
    mid[i] = bit[i]

# 模2除法主体
for m in range(0, A - 1):
    if mid[0] == 0:
        ans = 0

    elif mid[0] == 1:
        ans = 1
        for k in range(0, L + 1):
            mid[k] = mid[k] ^ g_24A[k]
    for n in range(0, L - 1):
        mid[n] = mid[n + 1]
    mid[L] = bit[m + L + 1]
# 判断最后一步还是否需要模2除法
if mid[0] == 0:
    ans = 0
elif mid[0] == 1:
    ans = 1
    for j in range(0, L + 1):
        mid[j] = mid[j] ^ g_24A[j]

for i in range(0, L):
    p[i] = mid[i + 1]

# 定义CRC编码后的数据帧为b
b = np.zeros(A + L + 1)
for i in range(0, L):
    bit[i + A] = p[i]
b = bit
print(b)
'''至此已实现CRC'''

'''第三步码块分割及CBCrc添加'''
# 送入分割的数据长度为B
B = A + L

graph = int(input('请输入graph号(1/2)'))
if graph == 1:
    K_cb = 8448
elif graph == 2:
    K_cb = 3840

if B <= K_cb:
    L = 0
    C = 1
    B_1 = B
else:
    L = 24
    C = math.ceil(B / (K_cb - L))
    B_1 = B + C * L
# 定义K‘为K_1
K_1 = math.ceil(B_1 / C)

if graph == 1:
    K_b = 22
elif graph == 2:
    if B > 640:
        K_b = 10
    elif B > 560:
        K_b = 9
    elif B > 192:
        K_b = 8
    else:
        K_b = 6

# 定义Z的最小值为Zc,寻找Zc
Z = np.matrix([[2, 4, 8, 16, 32, 64, 128, 256], [3, 6, 12, 24, 48, 96, 192, 384], [5, 10, 20, 40, 80, 160, 320, 0],
               [7, 14, 28, 56, 112, 224, 0, 0],
               [9, 18, 36, 72, 144, 288, 0, 0], [11, 22, 44, 88, 176, 352, 0, 0], [13, 26, 52, 104, 208, 0, 0, 0],
               [15, 30, 60, 120, 240, 0, 0, 0]])

# 定义中间矩阵Z_1，用于存放每一行满足条件的最小值
Z_1 = np.zeros(8)
for i in range(0, 8):
    for j in range(0, 8):
        while Z[i, j] * K_b >= K_1:
            Z_1[i] = Z[i, j]
            break
    if Z_1[i] == 0:
        Z_1[i] = 600

# Z_1中已存储每一行满足Z*Kb>=K'，取数组最小值即为Kc
Zc = int(min(Z_1))

if graph == 1:
    K = 22 * Zc
elif graph == 2:
    K = 10 * Zc

# 计算crk（即输出的分割后的比特流）
# 创建C*K'-L的零矩阵用于添加
c = np.zeros((C, K_1 - L))
# 定义余数p_c
p_c = np.zeros((C, L), dtype=int)
# 定义g_CRC24B生成函数
g_24B = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]
s = 0
# 给分块比特流后添加L位补位为模2除法做准备
zero_C = np.zeros(C)
for i in range(0, L):
    c = np.column_stack((c, zero_C))

for r in range(0, C):
    for k in range(0, K_1 - L):
        if s < B:
            c[r, k] = b[s]
            s = s + 1
        else:
            c[r, k] = 0
    if C > 1:

        # 定义中间变量mid_c与每一位计算结果ans_c
        mid_c = np.zeros((L + 1), dtype=int)
        ans_c = 0

        #############################模2除法

        for i in range(0, L + 1):
            mid_c[i] = c[r, i]

        # 模2除法主体
        for i in range(0, K_1 - L - 1):
            if mid_c[0] == 0:
                ans_c = 0

            elif mid_c[0] == 1:
                ans_c = 1
                for j in range(0, L + 1):
                    mid_c[j] = mid_c[j] ^ g_24B[j]
            for m in range(0, L - 1):
                mid_c[m] = mid_c[m + 1]
            mid_c[L] = c[r, i + L + 1]
        # 判断最后一步还是否需要模2除法
        if mid_c[0] == 0:
            ans_c = 0
        elif mid_c[0] == 1:
            ans_c = 1
            for n in range(0, L + 1):
                mid_c[n] = mid_c[n] ^ g_24B[n]

        for t in range(0, L):
            p_c[r, t] = mid_c[t + 1]

        for k in range(K_1 - L, K_1):
            c[r, k] = p_c[r, k + L - K_1]

zero_K = np.zeros(C)
for i in range(0, K - K_1):
    c = np.column_stack((c, zero_K))

for r in range(0, C):
    for k in range(K_1, K):
        c[r, k] = None

'''至此5.2.2节完毕'''

'''5.3.2部分给定的代码块的信道编码'''
'''取5.2.2节输出r=1的代码块，即第一个代码块为本节输入c_1'''
# 创建输入c_1即5.3.2中的ck
if C == 1:
    c_1 = c
else:
    c_1 = c[0, :]

if graph == 1:
    N = 66 * Zc
elif graph == 2:
    N = 50 * Zc

# 创建编码之后比特流为d
d = np.zeros(N)

# 步骤1
i_LS = 0
min_num = Z_1[0]
for i in range(len(Z_1)):
    if Z_1[i] < min_num:
        min_num = Z_1[i]
        i_LS = i

# 步骤2
for k in range(2 * Zc, K):
    if not math.isnan(c_1[i]):
        # if c_1[k] != None:
        d[k - 2 * Zc] = c_1[k]
    else:
        c_1[k] = 0
        d[k - 2 * Zc] = None

# 步骤3
# 创建奇偶校验位w
# w = np.zeros(N + 2 * Zc - K)
# w = w.T
c_1 = c_1.T

'''读取5.3.2-2与5.3.2-3表'''

dataFile = open("LDPCGraphValues.txt")
dataString = dataFile.read()

dataStringLines = dataString.splitlines()

lDPCGraphsValue = dict()

for line in dataStringLines:
    attr = line.split(",")
    lDPCGraphsValue[(attr[0], int(attr[1]), int(attr[2]), int(attr[3]))] = int(attr[4])


# 定义循环移位函数circshift,(被操作矩阵，向下移位数，向右移位数)
def circshift(matrix, shiftnum1, shiftnum2):
    h, z = matrix.shape
    matrix = np.vstack((matrix[(h - shiftnum1):, :], matrix[:(h - shiftnum1), :]))
    matrix = np.hstack((matrix[:, (z - shiftnum2):], matrix[:, :(z - shiftnum2)]))
    return matrix


# 计算得到移位个数p_i(i,j)
# 方便计算定义p_mid将p_i拉成向量
if graph == 1:

    p_i = np.zeros((46, 68))
    for i in range(0, 46):
        for j in range(0, 68):
            if ('table1', i, j, i_LS) in lDPCGraphsValue.keys():
                p_i[i, j] = lDPCGraphsValue[('table1', i, j, i_LS)] % Zc
            else:
                p_i[i, j] = -1
    p_mid = p_i.ravel()
    # 创建list存储变量名，方便矩阵拼接调用
    mat = []

    for m in range(0, 46 * 68):
        if p_mid[m] != -1:
            exec("Mat%s=circshift(np.eye(Zc),0 ,int(p_mid[m]))" % m)
            # exec("Mat%s=np.eye(Zc,dtype = int, k = int(p_mid[m]))" %m)
            mat.append("Mat%s" % m)
        elif p_mid[m] == -1:
            exec("Mat%s=np.zeros((Zc, Zc))" % m)
            mat.append("Mat%s" % m)

elif graph == 2:

    p_i = np.zeros((42, 52))
    for i in range(0, 42):
        for j in range(0, 52):
            if ('table2', i, j, i_LS) in lDPCGraphsValue.keys():
                p_i[i, j] = lDPCGraphsValue[('table2', i, j, i_LS)] % Zc
            else:
                p_i[i, j] = -1
    p_mid = p_i.ravel()
    mat = []

    for m in range(0, 42 * 52):
        if p_mid[m] != -1:
            exec("Mat%s=circshift(np.eye(Zc),0 ,int(p_mid[m]))" % m)
            # exec("Mat%s=np.eye(Zc,dtype = int, K = int(p_mid[m]))" %m)
            mat.append("Mat%s" % m)
        elif p_mid[m] == -1:
            exec("Mat%s=np.zeros((Zc, Zc))" % m)
            mat.append("Mat%s" % m)

'''已得到所有p(i,j)下的矩阵Mat，下来用矩阵的拼接成为H'''

# 横向拼接
if graph == 1:
    for m in range(0, 46):
        H = np.append(eval(mat[m * 68]), eval(mat[m * 68 + 1]), axis=1)

        for n in range(m * 68 + 2, m * 68 + 68):
            H = np.append(H, eval("Mat%s" % n), axis=1)
        exec("H%s = H" % m)

elif graph == 2:
    for m in range(0, 42):
        H = np.append(eval(mat[m * 52]), eval(mat[m * 52 + 1]), axis=1)

        for n in range(m * 52 + 2, m * 52 + 52):
            H = np.append(H, eval("Mat%s" % n), axis=1)
        exec("H%s = H" % m)

    # 纵向拼接
H = np.append(H0, H1, axis=0)
if graph == 1:
    for i in range(2, 46):
        H = np.append(H, eval("H%s" % i), axis=0)
elif graph == 2:
    for i in range(2, 42):
        H = np.append(H, eval("H%s" % i), axis=0)

'''至此已得到H'''

'''计算向量w'''

# 由于矩阵是非方阵的，因此为了方便计算，结合比特流c是已知的，因此改写矩阵
# 定义等式右边的向量为b_r
if graph == 1:
    m = 46 * Zc
    n = 68 * Zc
elif graph == 2:
    m = 42 * Zc
    n = 52 * Zc

b_r = np.zeros(m)
H_c = H[:, :K]

# 去掉None，以全零代替
for i in range(0, len(c_1)):
    if not math.isnan(c_1[i]):
        continue
    else:
        c_1[i] = 0

# c_mid = c_1.reshape(1, c_1.shape[0])

# c_1_T = c_mid.T
H_c = H_c.astype("int")
c_1 = c_1.astype("int")
b_r = b_r.astype("int")
for i in range(0, m):
    for j in range(0, K):
        b_r[i] = b_r[i] ^ H_c[i, j] * c_1[j]

b_r_mid = b_r.reshape(1, b_r.shape[0])
b_r = b_r_mid.T

H_w = H[:, K:n]
H_w = H_w.astype("int")

w = np.linalg.solve(H_w, b_r)

w = w.astype("int")
for i in range(0, m):
    w[i, 0] = abs(w[i, 0])
    if w[i, 0] in [0, 2, 4, 6, 8]:
        w[i, 0] = 0
    elif w[i, 0] in [1, 3, 5, 7, 9]:
        w[i, 0] = 1
    else:
        continue

'''步骤3已完成'''

'''步骤4开始'''
for k in range(K, N + 2 * Zc):
    d[k - 2 * Zc] = w[k - K, 0]

'''至此已得到LPDC编码'''
print(d)