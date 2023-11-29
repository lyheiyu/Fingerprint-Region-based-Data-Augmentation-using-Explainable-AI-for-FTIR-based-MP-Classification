'''
最小二乘法拟合函数曲线f(x)
1、拟合多项式为：y = a0 + a1*x + a2*x^2 + ... + ak*x^k
2、求每个点到曲线的距离之和：Loss = ∑(yi - (a0 + a1*x + a2*x^2 + ... + ak*x^k))^2
3、最优化Loss函数，即求Loss对a0,a1,...ak的偏导数为0
    3.1、数学解法——求解线性方程组：
        整理最优化的偏导数矩阵为：X：含有xi的系数矩阵，A：含有ai的系数矩阵，Y：含有yi的系数矩阵
        求解：XA=Y中的A矩阵
    3.2、迭代解法——梯度下降法：
        计算每个系数矩阵A[k]的梯度，并迭代更新A[k]的梯度
        A[k] = A[k] - (learn_rate * gradient)
'''
import numpy
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
'''
高斯列主消元算法
'''
# 得到增广矩阵
def get_augmented_matrix(matrix, b):
    row, col = np.shape(matrix)
    matrix = np.insert(matrix, col, values=b, axis=1)
    return matrix
# 取出增广矩阵的系数矩阵（第一列到倒数第二列）
def get_matrix(a_matrix):
    return a_matrix[:, :a_matrix.shape[1] - 1]
# 选列主元，在第k行后的矩阵里，找出最大值和其对应的行号和列号
def get_pos_j_max(matrix, k):
    max_v = np.max(matrix[k:, :])
    pos = np.argwhere(matrix == max_v)
    i, _ = pos[0]
    return i, max_v
# 矩阵的第k行后，行交换
def exchange_row(matrix, r1, r2, k):
    matrix[[r1, r2], k:] = matrix[[r2, r1], k:]
    return matrix
# 消元计算(初等变化)
def elimination(matrix, k):
    row, col = np.shape(matrix)
    for i in range(k + 1, row):
        m_ik = matrix[i][k] / matrix[k][k]
        matrix[i] = -m_ik * matrix[k] + matrix[i]
    return matrix
# 回代求解
def backToSolve(a_matrix):
    matrix = a_matrix[:, :a_matrix.shape[1] - 1]  # 得到系数矩阵
    b_matrix = a_matrix[:, -1]  # 得到值矩阵
    row, col = np.shape(matrix)
    x = [None] * col  # 待求解空间X
    # 先计算上三角矩阵对应的最后一个分量
    x[-1] = b_matrix[col - 1] / matrix[col - 1][col - 1]
    # 从倒数第二行开始回代x分量
    for _ in range(col - 1, 0, -1):
        i = _ - 1
        sij = 0
        xidx = len(x) - 1
        for j in range(col - 1, i, -1):
            sij += matrix[i][j] * x[xidx]
            xidx -= 1
        x[xidx] = (b_matrix[i] - sij) / matrix[i][i]
    return x
# 求解非齐次线性方程组：ax=b
def solve_NLQ(a, b):
    a_matrix = get_augmented_matrix(a, b)
    for k in range(len(a_matrix) - 1):
        # 选列主元
        max_i, max_v = get_pos_j_max(get_matrix(a_matrix), k=k)
        # 如果A[ik][k]=0，则矩阵奇异退出
        if a_matrix[max_i][k] == 0:
            print('矩阵A奇异')
            return None, []
        if max_i != k:
            a_matrix = exchange_row(a_matrix, k, max_i, k=k)
        # 消元计算
        a_matrix = elimination(a_matrix, k=k)
    # 回代求解
    X = backToSolve(a_matrix)
    return a_matrix, X
'''
最小二乘法多项式拟合曲线
'''
# 生成带有噪点的待拟合的数据集合
def init_fx_data():
    # 待拟合曲线f(x) = sin2x * [(x^2 - 1)^3 + 0.5]
    xs = np.arange(-1, 1, 0.01)  # 200个点
    ys = [((x ** 2 - 1) ** 3 + 0.5) * np.sin(x * 2) for x in xs]
    ys1 = []
    for i in range(len(ys)):
        z = np.random.randint(low=-10, high=10) / 100  # 加入噪点
        ys1.append(ys[i] + z)
    return xs, ys1
# 计算最小二乘法当前的误差
def last_square_current_loss(xs, ys, A):
    error = 0.0
    for i in range(len(xs)):
        y1 = 0.0
        for k in range(len(A)):
            y1 += A[k] * xs[i] ** k
        error += (ys[i] - y1) ** 2
    return error
# 迭代解法：最小二乘法+梯度下降法
def last_square_fit_curve_Gradient(xs, ys, order, iternum=1000, learn_rate=0.001):
    A = [0.0] * (order + 1)
    for r in range(iternum + 1):
        for k in range(len(A)):
            gradient = 0.0
            for i in range(len(xs)):
                y1 = 0.0
                for j in range(len(A)):
                    y1 += A[j] * xs[i]**j
                gradient += -2 * (ys[i] - y1) * xs[i]**k  # 计算A[k]的梯度
            A[k] = A[k] - (learn_rate * gradient)  # 更新A[k]的梯度
        # 检查误差变化
        if r % 100 == 0:
            error = last_square_current_loss(xs=xs, ys=ys, A=A)
            print('最小二乘法+梯度下降法：第{}次迭代，误差下降为：{}'.format(r, error))
    return A
# 数学解法：最小二乘法+求解线性方程组
import random
def last_square_fit_curve_Gauss(xs, ys, order):
    X, Y = [], []
    # 求解偏导数矩阵里，含有xi的系数矩阵X
    for i in range(0, order + 1):
        X_line = []
        for j in range(0, order + 1):
            sum_xi = 0.0
            for xi in xs:
                sum_xi += xi ** (j + i)
            X_line.append(sum_xi)
        X.append(X_line)
    # 求解偏导数矩阵里，含有yi的系数矩阵Y
    for i in range(0, order + 1):
        Y_line = 0.0
        for j in range(0, order + 1):
            sum_xi_yi = 0.0
            for k in range(len(xs)):
                sum_xi_yi += (xs[k] ** i * ys[k])
            Y_line = sum_xi_yi

        Y.append(Y_line)
    a_matrix, A = solve_NLQ(np.array(X), Y)  # 高斯消元：求解XA=Y的A
    #A = np.linalg.solve(np.array(X), np.array(Y))  # numpy API 求解XA=Y的A
    error = last_square_current_loss(xs=xs, ys=ys, A=A)

    print('最小二乘法+求解线性方程组，误差下降为：{}'.format(error))
    return A
# 可视化多项式曲线拟合结果
def generatedataBySperateLS(normalized_wns,intensity,n):
    order=4
    intensityforLoop=[]
    for u in range(len(intensity)):
        nstep=7
        m=int(len(intensity[u])/nstep)
        print(m)
        wavetotal = np.zeros(0)
        intensitytotal = np.zeros(0)
        restotal=np.zeros(0)
        for i in range(m):

            if i==m-1:
                wavestep = normalized_wns[i * nstep:]
                intensitystep = intensity[u][i * nstep:]

            else:
                wavestep=normalized_wns[i*nstep:(i+1)*nstep]
                intensitystep=intensity[u][i*nstep:(i+1)*nstep]
            A = last_square_fit_curve_Gauss(xs=wavestep, ys=intensitystep, order=order)
            #A = last_square_fit_curve_Gradient(xs=wavestep, ys=intensitystep, order=order, iternum=1000,
             #                                  learn_rate=0.001)
            wavestep, intensitystep, res = draw_fit_curve(xs=wavestep, ys=intensitystep, A=A, order=order,
                                                          intensity=intensity[u])
            wavetotal = np.concatenate((wavetotal, wavestep), axis=0)
            intensitytotal = np.concatenate((intensitytotal, intensitystep), axis=0)
        intensityforLoop.append(intensitytotal)
        restotal = np.concatenate((restotal, res), axis=0)
        y_add=[]
        for i in range(len(intensityforLoop)):
            y_add.append(n)
        #intensityforLoop=np.array(intensityforLoop)
        # print(intensityforLoop.shape)
    return intensityforLoop,y_add
def generatedataBySperateLSforEach(normalized_wns,intensity,n):
    order=4
    y_add=[]
    intensityforLoop=[]
    for u in range(len(intensity)):
        nstep=50
        m=int(len(intensity[u])/2)
        print(m)
        wavetotal = np.zeros(0)
        intensitytotal = np.zeros(0)
        restotal=np.zeros(0)
        for i in range(len(intensity)):

            intensityforLoop.append( np.concatenate((intensity[u][:m], intensity[i][m:]), axis=0))




    for i in range(len(intensityforLoop)):
            y_add.append(n)
    intensityforLoop=np.array(intensityforLoop)
        # print(intensityforLoop.shape)
    print(intensityforLoop.shape)
    return intensityforLoop,y_add
def generatedataBySperateLSforEach3(normalized_wns,intensity,n):
    order=4
    y_add=[]
    intensityforLoop=[]
    for u in range(len(intensity)):
        nstep=50
        m=int(len(intensity[u])/3)
        print(m)
        wavetotal = np.zeros(0)
        intensitytotal = np.zeros(0)
        restotal=np.zeros(0)
        for i in range(len(intensity)):
            for j in range(len(intensity)):

                temp=np.concatenate((intensity[u][:m], intensity[i][m:2*m]), axis=0)
                intensityforLoop.append(np.concatenate((temp,intensity[j][int(2*m):]), axis=0))


    for i in range(len(intensityforLoop)):
            y_add.append(n)
    intensityforLoop=np.array(intensityforLoop)
        # print(intensityforLoop.shape)
    print(intensityforLoop.shape)
    return intensityforLoop,y_add
def generatedataBySperateLSforEach4(wavelength,intensity,n):
    order=4
    y_add=[]
    wavelength=np.array(wavelength)
    index1=np.where(wavelength<1200)
    index1=np.min(index1)
    index2 = np.where(wavelength < 1900)
    print(index2)
    index2 = np.min(index2)
    index3 = np.where(wavelength < 2500)
    index3 = np.min(index3)

    print(index1,index2,index3)
    intensityforLoop=[]
    for u in range(len(intensity)):
        nstep=50
        m=int(len(intensity[u])/3)
        print(m)
        wavetotal = np.zeros(0)
        intensitytotal = np.zeros(0)
        restotal=np.zeros(0)
        for i in range(len(intensity)):
            for j in range(len(intensity)):
                for k in range(len(intensity)):

                    temp=np.concatenate((intensity[u][:index3], intensity[i][index3:index2]), axis=0)
                    temp=np.concatenate((temp,intensity[j][index2:index1]),axis=0)
                    intensityforLoop.append(np.concatenate((temp,intensity[k][index1:]), axis=0))


    for i in range(len(intensityforLoop)):
            y_add.append(n)
    intensityforLoop=np.array(intensityforLoop)
        # print(intensityforLoop.shape)
    print(intensityforLoop.shape)
    return intensityforLoop,y_add
def generatedataBySperateLSforEach5(wavelength,intensity,n):
    order=4
    y_add=[]
    wavelength=np.array(wavelength)
    index1=np.where(wavelength<1800)
    index1=np.min(index1)
    index2 = np.where(wavelength < 2500)
    index2 = np.min(index2)

    print(index1,index2)
    intensityforLoop=[]
    for u in range(len(intensity)):
        nstep=50
        m=int(len(intensity[u])/3)
        print(m)
        wavetotal = np.zeros(0)
        intensitytotal = np.zeros(0)
        restotal=np.zeros(0)
        for i in range(len(intensity)):
            for j in range(len(intensity)):


                temp=np.concatenate((intensity[u][:index2], intensity[i][index2:index1]), axis=0)
                #temp=np.concatenate((temp,intensity[j][index2:index1]),axis=0)
                intensityforLoop.append(np.concatenate((temp,intensity[j][index1:]), axis=0))


    for i in range(len(intensityforLoop)):
            y_add.append(n)
    intensityforLoop=np.array(intensityforLoop)
        # print(intensityforLoop.shape)
    print(intensityforLoop.shape)
    return intensityforLoop,y_add
def generatedataBySperateLSforEach6(wavelength,intensity,n):
    order=4
    y_add=[]
    wavelength=np.array(wavelength)
    # index01 = np.where(wavelength < 800)
    # index01 = np.min(index01)
    # index0 = np.where(wavelength < 1300)
    # index0=np.min(index0)
    index1=np.where(wavelength<1500)
    index1=np.min(index1)
    # index12 = np.where(wavelength < 2400)
    # index12 = np.min(index12)
    index2 = np.where(wavelength < 2900)
    index2 = np.min(index2)
    # index22 = np.where(wavelength < 3400)
    # index22 = np.min(index22)

    seprate1=[]
    seprate2=[]
    seprate3=[]
    # seprate4= []
    # seprate5 = []
    print(index1,index2)
    intensityforLoop=[]
    for nums in range(len(intensity)):
        # seprate1.append(intensity[nums][:index2])
        # seprate1.append(intensity[nums][:index2])
        seprate1.append(intensity[nums][:index2])
        seprate2.append(intensity[nums][index2:index1])
        seprate3.append(intensity[nums][index1:])
        # seprate4.append(intensity[nums][index0:index01])
        # seprate5.append(intensity[nums][index01:])
    ####for the 1st and second dataset
    #for mt in range(500):
    ####for the 4th dataset
    for mt in range(500):
        s1 = random.choice(seprate1)
        s2 = random.choice(seprate2)
        s3 = random.choice(seprate3)

        temp=np.concatenate((s1,s2),axis=0)
        temp=np.concatenate((temp,s3),axis=0)

        intensityforLoop.append(temp)
    for i in range(len(intensityforLoop)):
        y_add.append(n)
    intensityforLoop=np.array(intensityforLoop)
        # print(intensityforLoop.shape)
    return intensityforLoop,y_add
def generatedataBySperateLSforEach7(normalized_wns,intensity,n):
    order=4
    y_add=[]
    intensityforLoop=[]
    for u in range(len(intensity)):
        nstep=50
        m=int(len(intensity[u])/3)
        print(m)
        wavetotal = np.zeros(0)
        intensitytotal = np.zeros(0)
        restotal=np.zeros(0)
        seprate1 = []
        seprate2 = []
        seprate3 = []
        # seprate4= []
        # seprate5 = []

        intensityforLoop = []
        for nums in range(len(intensity)):
            # seprate1.append(intensity[nums][:index2])
            # seprate1.append(intensity[nums][:index2])
            seprate1.append(intensity[nums][:m])
            seprate2.append(intensity[nums][m:2*m])
            seprate3.append(intensity[nums][2*m:])
            # seprate4.append(intensity[nums][index0:index01])
            # seprate5.append(intensity[nums][index01:])

        for mt in range(20):
            s1 = random.choice(seprate1)
            s2 = random.choice(seprate2)
            s3 = random.choice(seprate3)

            temp = np.concatenate((s1, s2), axis=0)
            temp = np.concatenate((temp, s3), axis=0)

            intensityforLoop.append(temp)
        for i in range(len(intensityforLoop)):
            y_add.append(n)
        intensityforLoop = np.array(intensityforLoop)
        # print(intensityforLoop.shape)
        print(intensityforLoop.shape)
        return intensityforLoop, y_add
def generatedataBySperateLSforEach8(wavelength,intensity,n):
    order=4
    y_add=[]
    wavelength=np.array(wavelength)
    # index01 = np.where(wavelength < 800)
    # index01 = np.min(index01)
    # index0 = np.where(wavelength < 1300)
    # index0=np.min(index0)
    index1=np.where(wavelength < 1500)
    index1=np.min(index1)
    # index12 = np.where(wavelength < 2400)
    # index12 = np.min(index12)
    # index2 = np.where(wavelength < 2400)
    # index2 = np.min(index2)
    index2 = np.where(wavelength < 2700)
    index2 = np.min(index2)
    index3= np.where(wavelength < 3000)
    index3 = np.min(index3)
    # index22 = np.where(wavelength < 3400)
    # index22 = np.min(index22)

    seprate1=[]
    seprate2=[]
    seprate3=[]
    seprate4= []
    # seprate5 = []
    print(index1,index2,index3)
    intensityforLoop=[]
    for nums in range(len(intensity)):
        # seprate1.append(intensity[nums][:index2])
        # seprate1.append(intensity[nums][:index2])
        seprate1.append(intensity[nums][:index3])
        seprate2.append(intensity[nums][index3:index2])
        seprate3.append(intensity[nums][index2:index1])
        seprate4.append(intensity[nums][index1:])
        # seprate4.append(intensity[nums][index0:index01])
        # seprate5.append(intensity[nums][index01:])
    ####for the 1st and second dataset
    #for mt in range(500):
    ####for the 4th dataset
    for mt in range(1000):
        s1 = random.choice(seprate1)
        s2 = random.choice(seprate2)
        s3 = random.choice(seprate3)
        s4 = random.choice(seprate4)
        temp=np.concatenate((s1,s2),axis=0)
        temp=np.concatenate((temp,s3),axis=0)
        temp = np.concatenate((temp, s4), axis=0)
        intensityforLoop.append(temp)
    for i in range(len(intensityforLoop)):
        y_add.append(n)
    intensityforLoop=np.array(intensityforLoop)
    print(intensityforLoop.shape)
    return intensityforLoop,y_add
def generatedataBySperateLSforEach9(wavelength,intensity,n):
    order=4
    y_add=[]
    wavelength=np.array(wavelength)
    # index01 = np.where(wavelength < 800)
    # index01 = np.min(index01)
    # index0 = np.where(wavelength < 1300)
    # index0=np.min(index0)
    index1=np.where(wavelength<1700)
    index1=np.min(index1)
    # index12 = np.where(wavelength < 2400)
    # index12 = np.min(index12)
    index2 = np.where(wavelength < 3100)
    index2 = np.min(index2)
    # index22 = np.where(wavelength < 3400)
    # index22 = np.min(index22)

    seprate1=[]
    seprate2=[]
    seprate3=[]
    # seprate4= []
    # seprate5 = []
    print(index1,index2)
    intensityforLoop=[]
    for nums in range(len(intensity)):
        # seprate1.append(intensity[nums][:index2])
        # seprate1.append(intensity[nums][:index2])
        seprate1.append(intensity[nums][:index2])
        seprate2.append(intensity[nums][index2:index1])
        seprate3.append(intensity[nums][index1:])
        # seprate4.append(intensity[nums][index0:index01])
        # seprate5.append(intensity[nums][index01:])
    ####for the 1st and second dataset
    #for mt in range(500):
    ####for the 4th dataset
    for mt in range(500):
        s1 = random.choice(seprate1)
        s2 = random.choice(seprate2)
        s3 = random.choice(seprate3)

        temp=np.concatenate((s1,s2),axis=0)
        temp=np.concatenate((temp,s3),axis=0)

        intensityforLoop.append(temp)
    for i in range(len(intensityforLoop)):
        y_add.append(n)
    intensityforLoop=np.array(intensityforLoop)
        # print(intensityforLoop.shape)
    return intensityforLoop,y_add
def draw_fit_curve(xs, ys, A, order,intensity):
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    #fit_xs, fit_ys = np.arange(min(xs) , max(xs) , 0.01), []

    fit_ys=[]
    fit_xs=xs
    # for i in range(len(A)):
    #     A[i]=random.randint(10,20)*0.7*A[i]
    for i in range(0, len(fit_xs)):
        y = 0.0
        for k in range(0, order + 1):
            y += (A[k] * fit_xs[i] ** k)
        # A[0]=random.randint(0,int(A[0]))

        fit_ys.append(y)
    fit_ys=np.array(fit_ys)
    resid=ys-fit_ys


    # ax.plot(fit_xs, fit_ys, color='g', linestyle='-', marker='', label='poly-fit')
    # ax.plot(xs, ys, color='m', linestyle='', marker='.', label='true label')
    # #plt.title(s='least square'.format(order))
    # plt.legend()
    # plt.show()
    return fit_xs,fit_ys,resid
from utils import utils
from sklearn.model_selection import train_test_split
if __name__ == '__main__':
    order = 4  # 拟合的多项式项数

    xs, ys = init_fx_data()  # 曲线数据
    polymerName, waveLength, intensity, polymerID, x_each, y_each = utils.parseData2('D4_4_publication5.csv', 2, 1763)
    # MMScaler = MinMaxScaler()
    waveLength = np.array(waveLength)
    if waveLength[0] > waveLength[-1]:
        rng = waveLength[0] - waveLength[-1]
    else:
        rng = waveLength[-1] - waveLength[0]
    half_rng = rng / 2
    normalized_wns = (waveLength - np.mean(waveLength)) / half_rng
    # # ss = StandardScaler()
    # intensity=MMScaler.fit_transform(intensity)
    # LDA1=LinearDiscriminantAnalysis(n_components=10)
    # LDA2 = LinearDiscriminantAnalysis(n_components=10)
    # pca = PCA(n_components=0.99)
    # intensity=pca.fit_transform(intensity)

    x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.7, random_state=1)
    x = np.arange(0, 1000, 1)
    nstep=10
    m=int(len(intensity[0])/nstep)
    print(m)
    wavetotal = np.zeros(0)
    intensitytotal = np.zeros(0)
    restotal=np.zeros(0)

    # for i in range(m):
    #
    #
    #     wavestep=np.zeros(nstep)
    #     intensitystep=np.zeros(nstep)
    #     if i>=m-1:
    #         wavestep = normalized_wns[i * nstep:]
    #         intensitystep = intensity[0][i * nstep:]
    #
    #     else:
    #         wavestep=normalized_wns[i*nstep:(i+1)*nstep]
    #         intensitystep=intensity[0][i*nstep:(i+1)*nstep]
    #
    #     #print(wavestep,intensitystep)
    #     A = last_square_fit_curve_Gauss(xs=wavestep, ys=intensitystep, order=order)
    #     wavestep,intensitystep,res=draw_fit_curve(xs=wavestep, ys=intensitystep, A=A, order=order, intensity=intensity[0])
    #     wavetotal=np.concatenate((wavetotal, wavestep), axis=0)
    #     print(wavetotal.shape)
    #     intensitytotal = np.concatenate((intensitytotal, intensitystep), axis=0)
    #     restotal=np.concatenate((restotal,res),axis=0)
    # # restotal+=random.randint(-5,5)
    #
    # for muns in range(len(restotal)):
    #     if restotal[muns]>0.01:
    #         restotal[muns]+=random.randint(0,1)
    #
    #
    #
    # #wavetotal,intensitytotal=  draw_fit_curve(xs=wavetotal, ys=intensitytotal, A=A, order=order, intensity=intensity[0])
    # print(wavetotal.shape)
    # print(intensitytotal.shape)
    x,y=generatedataBySperateLSforEach6(waveLength,x_train[0:10],2)
    import numpy as np
    from pandas import read_csv
    import matplotlib.pyplot as plt
    from statsmodels.tsa.seasonal import seasonal_decompose
    from pylab import rcParams
    for m in range(len(x[0])):
        if  x[0][m]<=0:
            x[0][m]=0.01

    plt.clf()
    result = seasonal_decompose(x[0], model='multiplicative', freq=4)

    rcParams['figure.figsize'] = 10, 5
    result.plot()

    plt.figure(figsize=(40, 10))
    plt.show()
    x=numpy.array(x)
    print("x",x.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.plot(wavetotal, intensitytotal, color='g', linestyle='-', marker='', label='poly-fit')
    # ax.plot(wavetotal, restotal, color='b', linestyle='-', marker='', label='poly-fit')
    ax.plot(normalized_wns, x_train[0], color='m', linestyle='', marker='.', label='true label')
    for ti in range(len(x)):
        ax.plot(normalized_wns,x[ti], color='r', linestyle='-', marker='.', label='x label')
    #ax.plot(normalized_wns, x, color='r', linestyle='-', marker='.', label='x label')
    plt.show()
    waveLength = np.array(waveLength, dtype=np.float)
    # 数学解法：最小二乘法+求解线性方程组
    A = last_square_fit_curve_Gauss(xs=waveLength, ys=intensity[0], order=order)
    print(A)
    for i in range(len(A)):

        low= A[i]
        print(A[i])
        A[i]=0.01*random.randint(0,100)*A[i]
    print(A)
    # 迭代解法：最小二乘法+梯度下降
    #A = last_square_fit_curve_Gradient(xs=waveLength, ys=intensity[0], order=order, iternum=10000, learn_rate=0.001)
    draw_fit_curve(xs=waveLength, ys=intensity[0], A=A, order=order, intensity=intensity[0])  # 可视化多项式曲线拟合结果