from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import  recall_score
import pandas as pd
from PLS import airPLS
import  numpy as np
from sklearn.preprocessing import  normalize
import matplotlib.pyplot as plt
from sklearn.preprocessing import  MinMaxScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
class utils:
    def __init__(self):

        pass

    def printScore(y_true,y_pre):
        scores=[cohen_kappa_score,f1_score,accuracy_score,precision_score,recall_score]

        scoreList=[]
        for score in scores:
         # if score is f1, recall, accuracy set the average to macro
            if score.__name__ == 'recall_score' or score.__name__ == 'precision_score' or score.__name__ == 'f1_score':

                scoreList.append(score(y_true,y_pre,average='macro'))
            else: scoreList.append(score(y_true,y_pre))
            # print(score.__name__,score(y_true,y_pre))
        return scoreList
    def mkdir(path):
        # 判断目录是否存在
        # 存在：True
        # 不存在：False
        folder = os.path.exists(path)

        # 判断结果
        if not folder:
            # 如果不存在，则创建新目录
            os.makedirs(path)
            print('-----创建成功-----')

        else:
            # 如果目录已存在，则不创建，提示目录已存在
            print(path + '目录已存在')
    def parseData2(fileName, begin, end):
        dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False, low_memory=False)

        polymerName = dataset.iloc[1:, 1]
        waveLength = dataset.iloc[0, begin:end]
        intensity = dataset.iloc[1:, begin:end]
        polymerName = np.array(polymerName)
        intensity = np.array(intensity, dtype=np.float32)
        intensityBaseline = []
        for item in intensity:
            item = item - airPLS(item)
            intensityBaseline.append(item)

        # intensity=intensity-airPLS(intensity)

        # scaler= StandardScaler().fit(intensityBaseline)
        # intensity=scaler.transform(intensityBaseline)
        # minScaler = MinMaxScaler().fit(intensityBaseline)
        # intensity= minScaler.transform(intensityBaseline)
        intensity = normalize(intensityBaseline, 'max')
        #intensity = normalize(intensity, 'max')

        polymerID = dataset.iloc[1:, 1764]
        polymerID1=[]
        for item in polymerID:
            polymerID1.append(int(item)-1)
        polymerID = np.array(polymerID1)
        #polymerID= int(polymerID)-np.ones(polymerID.shape[0])

        x_class = []
        y_class = []
        for i in range(12):
            m = []
            z = []
            for j in range(len(intensity)):
                if int(polymerID[j]) == i:
                    z.append(polymerID[j])
                    m.append(intensity[j])
            x_class.append(m)
            y_class.append(z)

        return polymerName, waveLength, intensity, polymerID,x_class,y_class
    def parseData11(fileName, begin, end):
        dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False, low_memory=False)

        polymerName = dataset.iloc[1:, 1]
        waveLength = dataset.iloc[0, begin:end]
        intensity = dataset.iloc[1:, begin:end]
        polymerName = np.array(polymerName)
        intensity = np.array(intensity, dtype=np.float32)
        intensityBaseline = []
        for item in intensity:
            item = item - airPLS(item)
            intensityBaseline.append(item)

        # intensity=intensity-airPLS(intensity)

        # scaler= StandardScaler().fit(intensityBaseline)
        # intensity=scaler.transform(intensityBaseline)
        # minScaler = MinMaxScaler().fit(intensityBaseline)
        # intensity= minScaler.transform(intensityBaseline)
        intensity = normalize(intensityBaseline, 'max')
        #intensity = normalize(intensity, 'max')

        polymerID = dataset.iloc[1:, 1764]
        polymerID1=[]
        for item in polymerID:
            polymerID1.append(int(item)-1)
        polymerID = np.array(polymerID1)
        #polymerID= int(polymerID)-np.ones(polymerID.shape[0])

        x_class = []
        y_class = []
        for i in range(11):
            m = []
            z = []
            for j in range(len(intensity)):
                if int(polymerID[j]) == i:
                    z.append(polymerID[j])
                    m.append(intensity[j])
            x_class.append(m)
            y_class.append(z)

        return polymerName, waveLength, intensity, polymerID,x_class,y_class
    def plot_confusion_matrix(cm, labels_name, title):
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 25,
                 }
        font3 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 15,
                 }
        plt.imshow(cm, interpolation='nearest')  # 在特定的窗口上显示图像
        plt.title(title,font2)  # 图像标题
        plt.colorbar()
        ind_array = np.arange(len(labels_name))
        x, y = np.meshgrid(ind_array, ind_array)

        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm[y_val][x_val]
            if (c > 0.01):
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', va='center', ha='center',fontdict= font3)

        num_local = np.array(range(len(labels_name)))
        plt.xticks(num_local, labels_name, rotation=90,font=font2)  # 将标签印在x轴坐标上
        plt.yticks(num_local, labels_name,font=font2)  # 将标签印在y轴坐标上
        plt.ylabel('True label', font2)
        plt.xlabel('Predicted label', font2)
        plt.show()
    def parseDataForSecondDataset(fileName):
        dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False, low_memory=False)
        print(dataset)
        polymerName = dataset.iloc[1:, 0]
        waveLength = dataset.iloc[0, 1:-1]
        intensity = dataset.iloc[1:, 1:-1]
        polymerName = np.array(polymerName)
        intensity = np.array(intensity, dtype=np.float32)
        # intensity2 = []
        # for item in intensity:
        #     intensityeach = []
        #     item2 = item[::-1]
        #     for item3 in item2:
        #         if 100 - item3>=100:
        #             intensityeach.append(0)
        #
        #         else:
        #             intensityeach.append(100 - item3)
        #             print(max(intensityeach))
        #
        #     intensity2.append(intensityeach)
        intensity2=np.array(intensity)
        intensityBaseline = []
        for item in intensity:
            item = item - airPLS(item)
            intensityBaseline.append(item)
        # ss=StandardScaler()
        # intensity=ss.fit_transform(intensity)
        #intensity=intensity-airPLS(intensity)
        PN=[]
        for item in polymerName:
            if item not in PN:
                PN.append(item)
        # scaler= StandardScaler().fit(intensityBaseline)
        # intensity=scaler.transform(intensityBaseline)
        # minScaler = MinMaxScaler().fit(intensityBaseline)
        # intensity= minScaler.transform(intensityBaseline)
        intensity = normalize(intensityBaseline, 'max')
        #polymerID = dataset.iloc[1:, 1764]
        polymerID = dataset.iloc[1:, -1]
        polymerID1 = []
        for item in polymerID:
            polymerID1.append(int(item))
        polymerID = np.array(polymerID1)

        # polymerID1.append('polyID')
        # for item in polymerName:
        #     for i in range(len(PN)):
        #         if item==PN[i]:
        #             polymerID1.append(i)
        # print(polymerID1)
        # dataset['3552']=polymerID1
        # dataset.to_csv('new_SecondDataset.csv')

        #polymerID= int(polymerID)-np.ones(polymerID.shape[0])

        # x_class = []
        # y_class = []
        # for i in range(12):
        #     m = []
        #     z = []
        #     for j in range(len(intensity)):
        #         if int(polymerID[j]) == i:
        #             z.append(polymerID[j])
        #             m.append(intensity[j])
        #     x_class.append(m)
        #     y_class.append(z)

        return polymerName, waveLength, intensity, polymerID

    def parseDataForSecondDataset2(fileName):
        dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False, low_memory=False)
        # print(dataset)
        polymerName = dataset.iloc[1:, 0]
        waveLength = dataset.iloc[0, 1:-1]
        intensity = dataset.iloc[1:, 1:-1]
        polymerName = np.array(polymerName)
        intensity = np.array(intensity, dtype=np.float32)
        intensity2 = []
        for item in intensity:
            intensityeach = []
            item2 = item[::-1]
            for item3 in item2:
                if 100 - item3<=0:
                    intensityeach.append(0)
                    # print(100-item3)
                else:
                    intensityeach.append(100 - item3)


            intensity2.append(intensityeach)
        intensity2=np.array(intensity2)
        intensityBaseline = []
        for item in intensity2:
            item = item - airPLS(item)
            intensityBaseline.append(item)
        # ss=StandardScaler()
        # intensity=ss.fit_transform(intensity)
        #intensity=intensity-airPLS(intensity)
        PN=[]
        for item in polymerName:
            if item not in PN:
                PN.append(item)
        # scaler= StandardScaler().fit(intensityBaseline)
        # intensity=scaler.transform(intensityBaseline)
        # minScaler = MinMaxScaler().fit(intensityBaseline)
        # intensity= minScaler.transform(intensityBaseline)
        intensity = normalize(intensityBaseline, 'max')
        #polymerID = dataset.iloc[1:, 1764]
        polymerID = dataset.iloc[1:, -1]
        polymerID1 = []
        for item in polymerID:
            polymerID1.append(int(item))
        polymerID = np.array(polymerID1)

        # polymerID1.append('polyID')
        # for item in polymerName:
        #     for i in range(len(PN)):
        #         if item==PN[i]:
        #             polymerID1.append(i)
        # print(polymerID1)
        # dataset['3552']=polymerID1
        # dataset.to_csv('new_SecondDataset.csv')

        #polymerID= int(polymerID)-np.ones(polymerID.shape[0])

        # x_class = []
        # y_class = []
        # for i in range(12):
        #     m = []
        #     z = []
        #     for j in range(len(intensity)):
        #         if int(polymerID[j]) == i:
        #             z.append(polymerID[j])
        #             m.append(intensity[j])
        #     x_class.append(m)
        #     y_class.append(z)

        return polymerName, waveLength, intensity, polymerID
    def getPN(fileName):
        dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False, low_memory=False)

        polymerName = dataset.iloc[1:, 1]
        PN = []
        for item in polymerName:
            if item not in PN:
                PN.append(item)
        return PN

    def parseDataForBayes(fileName,fileName2):
        dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False, low_memory=False)
        dataset2 = pd.read_csv(fileName2, header=None, encoding='latin-1', keep_default_na=False, low_memory=False)
        recommendData=dataset2.iloc[1:,1:]
        recommendData=np.array(recommendData)
        # polymerID1=dataset.iloc[1:,-7]
        BayesClass=dataset.iloc[1:,-6:]
        BayesClass=np.array(BayesClass)

        for i in range(len(BayesClass)):
            for j in range(len(BayesClass[i])):
                #print(BayesClass[i][j])
                BayesClass[i][j]=int(BayesClass[i][j])
        for k in range(len(recommendData)):
            for l in range(len(recommendData[k])):

                    # print(BayesClass[i][j])
                recommendData[k][l] = int(recommendData[k][l])
            #BayesClass[i]=int(BayesClass[i])
        # polymerID1.append('polyID')
        # for item in polymerName:
        #     for i in range(len(PN)):
        #         if item==PN[i]:
        #             polymerID1.append(i)
        # print(polymerID1)
        # dataset['3552']=polymerID1
        # dataset.to_csv('new_SecondDataset.csv')
        polymerID = dataset.iloc[1:, -7]
        polymerID1 = []
        for item in polymerID:
            polymerID1.append(int(item) - 1)
        polymerID = np.array(polymerID1)
        #polymerID= int(polymerID)-np.ones(polymerID.shape[0])

        # x_class = []
        # y_class = []
        # for i in range(12):
        #     m = []
        #     z = []
        #     for j in range(len(intensity)):
        #         if int(polymerID[j]) == i:
        #             z.append(polymerID[j])
        #             m.append(intensity[j])
        #     x_class.append(m)
        #     y_class.append(z)

        return  polymerID,BayesClass,recommendData

    def parseData3rd(fileName):
        # dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False)
        dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False)
        # dataset = dataset.replace({'NaN': pd.np.nan, 'nan': pd.np.nan})
        # dataset= dataset.fillna(0)
        polymerName = dataset.iloc[1:172, -2]
        polymerID = dataset.iloc[1:172, -1]
        # waveLength=dataset.iloc[0,begin:end]
        intensity = dataset.iloc[1:172, 1:-2]

        intensity= intensity.replace([np.inf, -np.inf], 0)

        polymerID = np.array(polymerID,dtype=np.int)

        waveLength = dataset.iloc[0, 1:-2]
        # intensity = dataset.iloc[1:971, begin:end]
        intensity = np.array(intensity)
        for j in range(len(intensity)):

            for i in range(len(intensity[j])):
                if intensity[j][i] == '':
                    intensity[j][i] = 0.000000000

        intensity = np.array(intensity, dtype=np.float64)
        for j in range(len(intensity)):
            for i in range(len(intensity[j])):
                #intensity[j][i]= intensity[j][i].astype(np.float64)
                intensity[j][i] = float(intensity[j][i])
        for j in range(len(intensity)):
            for i in range(len(intensity[j])):
                #intensity[j][i]= intensity[j][i].astype(np.float64)
                intensity[j][i] = float(intensity[j][i])
                #print(type(intensity[j][i]))
                if type(intensity[j][i]) is not np.float64:
                    print(intensity[j][i])
                    print(1)
        intensity=np.nan_to_num(intensity)
        intensity[np.isnan(intensity)] = 0


        intensity[np.isinf(intensity)] = 0

        # scaler= StandardScaler().fit(intensity)
        # intensity=scaler.transform(intensity)
        # minScaler = MinMaxScaler().fit(intensity)
        # intensity= minScaler.transform(intensity)
        intensity = normalize(intensity, 'max')
        #intensity = normalize(intensity, 'max')
                # print(it.__class__)
            # print(item)
        polymerName = np.array(polymerName)
        waveLength = np.array(waveLength)
        for i in range(len(waveLength)):
            waveLength[i] = float(waveLength[i])
        return polymerName, waveLength, intensity, polymerID
    def parseData4th(fileName):
        # dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False)
        dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False)
        # dataset = dataset.replace({'NaN': pd.np.nan, 'nan': pd.np.nan})
        # dataset= dataset.fillna(0)
        polymerName = dataset.iloc[1:, -2]
        polymerID = dataset.iloc[1:, -1]
        # waveLength=dataset.iloc[0,begin:end]
        intensity = dataset.iloc[1:, :-2]

        # intensity= intensity.replace([np.inf, -np.inf], 0)

        polymerID = np.array(polymerID,dtype=np.int32)
        waveLength = dataset.iloc[0, :-2]
        intensity = np.array(intensity)
        waveLength = waveLength[::-1]
        intensity2 = []
        for item in intensity:
            intensityeach=[]
            item2 = item[::-1]
            for item3 in item2:
                if 100-item3>=100:
                    intensityeach.append(0)
                else: intensityeach.append(100-item3)

            intensity2.append(intensityeach)
        intensity2=np.array(intensity2)
        intensityBaseline = []
        for item in intensity2:
            item = item - airPLS(item)
            intensityBaseline.append(item)
        # intensity = np.array(intensity2)
        # scaler= StandardScaler().fit(intensity)
        # intensity=scaler.transform(intensity)
        # minScaler = MinMaxScaler().fit(intensity)
        # intensity= minScaler.transform(intensity)
        # intensity = normalize(intensity, 'max')
        intensity = normalize(intensityBaseline, 'max')
                # print(it.__class__)
            # print(item)
        polymerName = np.array(polymerName)
        waveLength = np.array(waveLength)
        for i in range(len(waveLength)):
            waveLength[i] = float(waveLength[i])
        return polymerName, waveLength, intensity, polymerID
if __name__ == '__main__':
    ut=utils
    #polymerName, waveLength, intensity, polymerID=ut.parseDataForSecondDataset('new_SecondDataset.csv')
    # print(intensity)
    # print(polymerID)
    # SVM_clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    #                   decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
    #                   max_iter=-1, probability=False, random_state=None, shrinking=True,
    #                   tol=0.001, verbose=False)
    # x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.3, random_state=1)
    # SVM_clf.fit(x_train,y_train)
    # print(SVM_clf.score(x_test,y_test))
    #polymerID, BayesClass,rec = ut.parseDataForBayes('D4_4_publication5for bayes.csv','totalPreForBayes.csv')
    polymerName, waveLength, intensity, polymerID = ut.parseData4th('dataset/FourthdatasetFollp-r.csv')
    print(waveLength)
    PN=[]
    for item in polymerName:
        if item not in PN:
            PN.append(item)
    eachLength=[]
    for n in range(len(PN)):
        numSynth = 2
        indicesPS = [l for l, id in enumerate(polymerID) if id == n]
        intensityForLoop = polymerID[indicesPS]
        eachLength.append(len(intensityForLoop))
    print(len(eachLength))
    print(len(PN))
    el=[]
    el.append(eachLength)
    el.append(PN)
    # eachLength=np.array(eachLength)
    # PN=np.array(PN)
    # el=np.concatenate((eachLength,PN),axis=1)
    el=pd.DataFrame(el)



    print(el)
    el.to_csv('dataset/info.csv')
    # waveLength=waveLength[::-1]
    # intensity2=[]
    # for item in intensity:
    #     item=item[::-1]
    #     intensity2.append(item)
    # intensity2=np.array(intensity2)
    # print(intensity2)
    # for item in intensity:
    #     print(item)
    # for item in intensity:
    #     for item2 in item:
    #         print(item2)
    #print(intensity)


    # print(BayesClass)
    # print(polymerID)
    # print(np.array(rec).shape)
    # print(rec)
