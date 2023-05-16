import keras.layers
from keras.layers import Dense, Input,Reshape
from keras.models import Model,load_model
import numpy as np
import pandas  as  pd
import seaborn as sns
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from sklearn.neural_network import  MLPClassifier
import  torch
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
MMScaler = MinMaxScaler()
from utils import utils
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from math import isnan
from sklearn.model_selection import cross_val_score
from sklearn import  svm
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, Conv1D,\
    MaxPool1D,AveragePooling1D,AveragePooling2D,GlobalAveragePooling1D
from keras.optimizer_v1 import SGD
from sklearn import metrics
from keras.utils import np_utils
from sklearn.cluster import KMeans
class TestKeras:
    def __init__(self):
        pass
    def getPN(fileName):
        dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False, low_memory=False)

        polymerName = dataset.iloc[1:, 1]
        PN = []
        for item in polymerName:
            if item not in PN:
                PN.append(item)
        return PN
    def plot_confusion_matrix(cm, labels_name, title):
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
        plt.imshow(cm, interpolation='nearest')  # 在特定的窗口上显示图像
        plt.title(title)  # 图像标题
        plt.colorbar()
        num_local = np.array(range(len(labels_name)))
        plt.xticks(num_local, labels_name, rotation=90)  # 将标签印在x轴坐标上
        plt.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    def parseData2(fileName,begin,end):

        dataset = pd.read_csv(fileName, header=None, encoding='latin-1',keep_default_na=False)
        polymerName=dataset.iloc[1:971,1]
        waveLength=dataset.iloc[0,begin:end]
        intensity=dataset.iloc[1:971,begin:end]
        polymerName= np.array(polymerName)
        waveLength=np.array(waveLength)
        #polymerNameList = {}.fromkeys(polymerName).keys()
        polymerNameList=[]
        pList= []
        for item in polymerName:
            if item not in pList:
                pList.append(item)

        polymerNameList=np.array(pList)
        polymerNameID=[]
        for i in range(len(polymerNameList)):
            polymerNameID.append(i+1)
        polymerNameData=[]
        for item in polymerName:
            for i in range(len(pList)):
                if item==pList[i]:
                    polymerNameData.append(i)


        intensity =np.array(intensity)
        for item in intensity:

            for i in range(len(item)):
                if item[i] == '':
                    item[i] = 0
        intensity = MMScaler.fit_transform(intensity)
        return polymerName,waveLength,intensity,polymerNameData
    def spectrumSVM(x,y,cvalue,kModel,decFunction):
        model=svm.SVC(C=cvalue, kernel=kModel, decision_function_shape=decFunction)

        model.fit(x,y)

        return model

    def getPN(fileName):
        dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False, low_memory=False)

        polymerName = dataset.iloc[1:, 1]
        PN = []
        for item in polymerName:
            if item not in PN:
                PN.append(item)
        return PN

    def plot_confusion_matrix(cm, labels_name, title):
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
        plt.imshow(cm, interpolation='nearest')  # 在特定的窗口上显示图像
        plt.title(title)  # 图像标题
        plt.colorbar()
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 20,
                 }
        num_local = np.array(range(len(labels_name)))
        plt.xticks(num_local, labels_name, rotation=90, size=15)  # 将标签印在x轴坐标上
        plt.yticks(num_local, labels_name, size=15)  # 将标签印在y轴坐标上
        plt.ylabel('True label', font2)
        plt.xlabel('Predicted label', font2)
        plt.show()
    def build_generateor(kernalsize):
        model = Sequential()
        model.add(Conv1D(1, kernalsize, activation='relu', input_shape=(len(intensity[0]), 1), padding="same"))
        model.add(MaxPool1D(pool_size=3, strides=3))
        model.add(Conv1D(1, 3, strides=1, activation='relu', padding='same'))
        model.add(MaxPool1D(pool_size=3, strides=3))
        model.add(Conv1D(1, 3, strides=1, activation='relu', padding='same'))
        model.add(MaxPool1D(pool_size=3, strides=3))
        return model

from sklearn.preprocessing import OneHotEncoder
if __name__ == '__main__':

    tS=TestKeras
    #this is for the 216_2018.csv
    #polymerName, polymerID, waveLength, intensity=tk.parseData2('216_2018_1156_MOESM5_ESM.csv',3,1179)
    #this is for the D4 public_csv
    polymerName, waveLength, intensity, polymerID, x_each, y_each = utils.parseData11('data/D4_4_publication11.csv', 2,1763)
    #polymerName, waveLength, intensity, polymerID = utils.parseDataForSecondDataset('data/new_SecondDataset.csv')
    #polymerName, waveLength, intensity, polymerID = utils.parseDataForSecondDataset('data/new_SecondDataset2.csv')
    ohe = OneHotEncoder()
    polymerID = polymerID.reshape(-1, 1)
    ohe.fit(polymerID)
    polymerID2 = ohe.transform(polymerID).toarray()
    for item in polymerID2:
        print(item)
    pList = list(set(polymerName))
    PN=[]
    for item in polymerName:
        if item not in PN:
            PN.append(item)
    print(max(polymerID))
    x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.3, random_state=2)
    input_img=1761
    encoding_dim=256
    datas=[]
    datas2=[]
    for n in range(len(PN)):
        numSynth = 2
        #indicesPS = [l for l, id in enumerate(y_train) if np.argmax(id) == n]
        indicesPS = [l for l, id in enumerate(y_train) if id == n]
        intensityForLoop = x_train[indicesPS]
        datas.append(intensityForLoop)
        datas2.append(intensityForLoop)
    input_img = Input(shape=(len(intensity[0]),))
    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    # y_train = np.array(y_train, dtype=np.float32)
    # y_test = np.array(y_test, dtype=np.float32)

    nb_features=1761
    nb_class=len(PN)
    print(nb_class)
    x_train=x_train.reshape(-1,len(intensity[0]),1)
    x_test = x_test.reshape(-1, len(intensity[0]), 1)
    print(x_train.shape)
    # X_train_r = np.zeros((len(x_train), nb_features, 3))
    # X_train_r[:, :, 0] = x_train[:, :nb_features]
    # X_train_r[:, :, 1] = x_train[:, nb_features:1024]
    # X_train_r[:, :, 2] = x_train[:, 1024:]
    # X_train_r = np.zeros((len(x_train), 1, 3))
    # X_train_r[:, :, 0] = x_train[:, :nb_features]
    # X_train_r[:, :, 1] = x_train[:, nb_features:1024]
    # X_train_r[:, :, 2] = x_train[:, 1024:]
    #feature = layer_model.predict(x_train)
    #x_test_feature= layer_model.predict(x_test)
    dense_num = 6
    # model=tS.build_generateor(64)
    #
    # # 神经元随机失活
    # #model.add(Dropout(0.25))
    # # 拉成一维数据
    # model.add(Flatten())
    # # # 全连接层1
    # model.add(Dense(len(intensity[0]),name='explainLayer'))
    # # 激活层
    # model.add(Activation('tanh'))
    filter=64
    model = Sequential()
    model.add(Reshape((len(intensity[0]), 1), input_shape=(len(intensity[0]),)))
    model.add(Conv1D(filter, 64, activation='relu', input_shape=(len(intensity[0]),1), padding="same"))
    model.add(MaxPool1D(pool_size=3, strides=3))
    model.add(Conv1D(filter, 64, strides=1, activation='relu', padding='same'))
    model.add(MaxPool1D(pool_size=3, strides=3))
    model.add(Conv1D(filter, 64, strides=1, activation='relu', padding='same'))
    #model.add(MaxPool1D(pool_size=1, strides=1))
    #model.add(AveragePooling1D(pool_size=1, strides=1))
    model.add(GlobalAveragePooling1D(data_format='channels_last'))
    #model.add(Conv1D(1, 1, strides=1, activation='relu', padding='same'))
    #model.add(Dropout(0.25))
    # 拉成一维数据
    model.add(Flatten())
    # 全连接层1

    model.add(Dense(len(PN),activation='softmax'))
    # Softmax评分
    # model.add(Activation('softmax'))

    # 查看定义的模型
    model.summary()

    # 自定义优化器参数
    # rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # lr表示学习速率
    # decay是学习速率的衰减系数(每个epoch衰减一次)
    # momentum表示动量项
    # Nesterov的值是False或者True，表示使不使用Nesterov momentum
    #sgd = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #
    # # 训练
    # # history = Models.fit(x_train, y_train, epochs=50, batch_size=16,
    # #                     verbose=1, validation_data=[x_test, y_test])
    #model.fit(x_train, y_train, epochs=50, batch_size=32,verbose=1)
    # y_pre=model.predict(x_test)
    # from sklearn.decomposition import IncrementalPCA
    #
    # ipca = IncrementalPCA(n_components=2, batch_size=3)
    # ipca.fit(X)
    # IncrementalPCA(batch_size=3, n_components=2)
    # ipca.transform(X)
    model_file= 'Models/Conv1D_FTIR_modelFirstdataset2.h5'
    #model.save(model_file)
    model=load_model(model_file)
    y_pre_index=[]
    y_pre = model.predict(x_test)
    for i in range(len(y_pre)):
        y_pre_index.append(np.argmax(y_pre[i]))
    y_pre_index2=np.argmax(y_pre,axis=1)
    print(y_pre_index2)
    # y_test_index=[]
    # for i in range(len(y_test)):
    #     y_test_index.append(np.argmax(y_test[i]))

    # utils.printScore(y_test_index,y_pre_index)
    # cm = confusion_matrix(y_test_index, y_pre_index)
    #
    # utils.plot_confusion_matrix(cm, PN, 'CNN')
    #
    # print(metrics.r2_score(y_test_index,y_pre_index))
    #autoencoder = Model(inputs=model.input, outputs=model.get_layer('explainLayer').output)
    cam = Model(inputs=model.input, outputs=model.layers[-4].output)

    weights = model.layers[-1].get_weights()[0]
    weights = np.array(weights)
    # print('weights shape', weights.shape)
    # print(intensity[0].shape)
    # for item in weights:
    #     print('weight each shape', item.shape)
    generate = []
    datas = np.array(datas)
    dataforeach = []
    for i in range(len(PN)):
        dataforeach.append(datas[i][0:4])
    dataforeach = np.array(dataforeach)
    generate2 = []
    for i in range(len(dataforeach)):
        generate2.append(cam.predict(dataforeach[i]))
        camp=np.array(cam.predict(dataforeach[i]))
        print(camp.shape)
    # generate2=cam.predict(dataforeach)
    generate2=np.array(generate2)
    # for i in range(len(dataforeach)):
    #     generate2[i, :] = savgol_filter(generate2[i, :], 207, 3, mode='nearest')
    map = []
    resultsforcluster=[]
    i=0
    gen4=[]
    for i in range(len(generate2)):
        genforeach=[]
        for j in range(len(generate2[0])):
            genforeach.append(generate2[i][j])
        gen4.append(genforeach)
    gen4=np.array(gen4)
    print('gen4',gen4.shape)

    print(weights.shape)
    print(generate2.shape)
    i=0
    newdata=[]

    # for j in range(len(PN)):
    #     newdata1=[]
    #     for k in range(len(generate2[j])):
    #         newdataeach = []
    #         for m in range(len(generate2[j][k])):
    #             for i in range(filter):
    #                 newdataeach.append(generate2[j][k][m][filter])
    #         newdata1.append(newdataeach)
    # #gen3=generate2[0][0].T
    # print(gen3.shape)
    ModifiedWeights=[]
    for i in range(len(PN)):
        modifyforeach=[]
        for item in weights:
            modifyforeach.append(item[i])
        ModifiedWeights.append(modifyforeach)

    ModifiedWeights=np.array(ModifiedWeights)
    print(ModifiedWeights.shape)
    heatmap4=[]
    heatmap5=[]
    heatmap2= []
    heatmap3 = []
    clusterdata=[]
    i=0
    for i in range(len(gen4)):
        mapeach = []
        eachcluser = []
        eachcc = []
        eachm = []
        for j in range(len(gen4[i])):
            # for j in range(2):
            iterateCluster = []

            xmtp =np.dot(gen4[i][j], ModifiedWeights[i])
            print(xmtp.shape)
            #iterateCluster.append([ModifiedWeights[i][j], gen4[i][j][m]])
            eachm.append(xmtp)
            # eachcc.append(eachm)
            # eachcluser.append(iterateCluster)

        clusterdata.append(eachcluser)
        heatmap4.append(eachm)
        heatmap5.append(eachm)
    heatmap4=np.array(heatmap4)
    print(heatmap4.shape)
    # for i in range(len(generate2)):
    #     mapeach = []
    #     eachcluser=[]
    #     eachcc = []
    #     for j in range(len(generate2[i])):
    #     # for j in range(2):
    #         iterateCluster=[]
    #         eachm=[]
    #         for m in range(len(generate2[i][j])):
    #             # print(ModifiedWeights[i][m],generate2[i][j][m])
    #             xmtp=ModifiedWeights[i][m]*generate2[i][j][m]
    #             # print(xmtp)
    #             iterateCluster.append([ModifiedWeights[i][m],generate2[i][j][m]])
    #             eachm.append(xmtp)
    #         eachcc.append(eachm)
    #         eachcluser.append(iterateCluster)
    #
    #     clusterdata.append(eachcluser)
    #     heatmap2.append(eachcc)
    #     heatmap3.append(eachcc)
    clusterdata=np.array(clusterdata)
    xspace = np.linspace(int(max(waveLength)), int(min(waveLength)), len(heatmap4[2][0]))
    i=0
    clusterdata1=[]
    heatforloop=heatmap4[3][0]
    for i in range(len(heatmap4[9][0])):
        clusterdata1.append([xspace[i],heatforloop[i]])
    clusterdata1=np.array(clusterdata1)
    print('clustershape', clusterdata1.shape)
    i=0
    minc=min(heatforloop)
    maxc=max(heatforloop)
    for i in range(len(clusterdata1)):
        clusterdata1[i][1]=  (clusterdata1[i][1]-minc)/(maxc-minc)

    from sklearn.mixture import GaussianMixture as GMM
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import gridspec
    print(len(heatmap4[2][0]))
    firstIndex=int(len(heatmap4[2][0])/3)
    secondIndex=2*firstIndex
    labels1=[]
    colors = ['c', 'b', 'g', 'r', 'm', 'y', '#377eb8', 'darkviolet', 'olive',
              'tan', 'lightgreen', 'gold', 'cyan', 'magenta', 'pink',
              'crimson', 'navy', 'cadetblue', '#ffffff', '#bbbbbb', '#aaaaaa', '#dddddd', '#123123', '#321321',
              '#533ef1', '#abeabe', 'c', 'b', 'g', 'r', 'm', 'y']
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20, }
    i=0
    for i in range(firstIndex):
        labels1.append(0)
    for i in range(firstIndex,secondIndex):
        labels1.append(1)
    for i in range(secondIndex,len(heatmap4[2][0])):
        labels1.append(2)
    gmm=GMM(n_components=3).fit(clusterdata1)
    labels=gmm.predict(clusterdata1)
    print(labels)
    # for item in labels:
    #     print(item)
    # print(labels)

    kmeanforspectrum=KMeans(n_clusters=3,random_state=200).fit(clusterdata1)
    labelkmeans=kmeanforspectrum.labels_
    plt.tick_params(labelsize=20)
    plt.scatter(clusterdata1[:,0],clusterdata1[:,1],c=labels)
    plt.title('GaussianMixture cluster',font2)
    plt.xlim(max(waveLength)+100, 400)
    plt.show()
    plt.tick_params(labelsize=20)
    plt.scatter(clusterdata1[:, 0], clusterdata1[:, 1], c=labelkmeans)
    plt.title('Kmeans cluster', font2)
    plt.xlim(max(waveLength) + 100, 400)
    plt.show()
    plt.scatter(clusterdata1[:, 0], clusterdata1[:, 1], c=labels1)
    plt.tick_params(labelsize=20)
    plt.xlim(max(waveLength)+100, 500)
    plt.title('True label cluster',font2)
    plt.show()
    # rsc=pd.DataFrame(clusterdata)
    # rsc.to_csv('heatmap.csv')
    # generate = []
    # datas = np.array(datas)
    # from sklearn.preprocessing import StandardScaler
    #
    # dataforeach = []
    # set=[0,1]
    # for i in set:
    #     dataforeach.append(datas[i][2])
    # dataforeach = np.array(dataforeach)
    # generate2 = autoencoder.predict(dataforeach)
    # generate2 = np.array(generate2)
    # SS = StandardScaler()
    # generate2 = SS.fit_transform(generate2)
    # # for i in range(len(PN)):
    # #    generate.append(autoencoder.predict([datas[i][0]]))
    # print(generate2.shape)

    heatmap2 = np.array(heatmap2)
    heatmap3 = np.array(heatmap3)
    print(heatmap2.shape)
    # for i in range(len(PN)):
    #     heatmap2[i, :] = savgol_filter(heatmap2[i, :], 301, 2, mode='nearest')
    #     heatmap3[i, :] = savgol_filter(heatmap3[i, :], 301, 2, mode='nearest')
    m=0
    u=0
    for m in range(len(heatmap4)):
        for u in range(len(heatmap4[m])):
            maxr = max(heatmap4[m][u])
            minr = min(heatmap4[m][u])
            en=0
            for en in range(len(heatmap4[m][u])):
                heatmap4[m][u][en] = (heatmap4[m][u][en] - minr) / (maxr - minr)
                heatmap5[m][u][en] = (heatmap5[m][u][en] - minr) / (maxr - minr)

    colors = ['c', 'b', 'g', 'r', 'm', 'y', '#377eb8', 'darkviolet', 'olive',
              'tan', 'lightgreen', 'gold', 'cyan', 'magenta', 'pink',
              'crimson', 'navy', 'cadetblue', '#ffffff', '#bbbbbb', '#aaaaaa', '#dddddd', '#123123', '#321321',
              '#533ef1', '#abeabe', 'c', 'b', 'g', 'r', 'm', 'y']
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20, }
    # map2=map

    mapforsean = heatmap5[9][1].reshape(-1, 1)
    i=0
    for i in range(len(mapforsean)):
        if mapforsean[i]>0.8:
            mapforsean[i]=1
        if 0.8>mapforsean[i]>0.6:
            mapforsean[i]=0.7
    sns.heatmap(mapforsean)
    from collections import OrderedDict
    import matplotlib.pyplot as plt


    # #plt.ylim(intensity.min(), intensity.max())
    fig, ax = plt.subplots()
    # x,y=fitCurve(waveLength,intensity3)
    # print(x.shape,y.shape)
    # plt.tick_params
    ax.tick_params(labelsize=20)
    plt.xlim(max(waveLength), 500)
    #plt.ylim(0, 1, font2)
    # print(polymerID)
    sets = [1,3,4,8,9]
    xspace=np.linspace(max(waveLength),min(waveLength),len(heatmap4[2][0]))
    for ni in range(len(sets)):
        # indicesPS = [l for l, id in enumerate(polymerID) if id == ni]
        # intensityloop = intensity[indicesPS]
        # print(len(intensityloop))
        # for mi in range(len(intensityloop)):
        #ax.plot(waveLength, generate2[ni], color=colors[ni], label=PN[set[ni]])
        for mi in range(len(heatmap4[sets[ni]])):
            # ax.plot(waveLength, map[ni][mi], color=colors[ni], label=PN[ni])
            ax.plot(xspace, heatmap4[sets[ni]][mi], color=colors[ni], label=PN[sets[ni]])
    # wandb.log({'Reward': intensity[0]}
    # wandb.log({'Reward': episode_reward})

    # ax.plot(wavenumber, HDPEintensity, color='r', label='Standard polyethylene')
    # plt.legend(labels=polymerNameSet2[0:2])
    # for j in range(len(y)):
    #     for i in range(len(polymerNameSet)):
    #         if polymerID[j] == str(i + 1):
    #              #print(polymerID[j])
    #             ax.plot(x, y[j], color=colors[i])
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=20)
    plt.xlabel('Wavelength', font2)
    plt.ylabel('Intensity', font2)
    plt.title('Explainable network for main MPs', font2)
    plt.show()

    utils.printScore(y_test, y_pre_index2)
    cm = confusion_matrix(y_test, y_pre_index2)

    utils.plot_confusion_matrix(cm, PN, 'CNN')

    print(metrics.r2_score(y_test, y_pre_index2))