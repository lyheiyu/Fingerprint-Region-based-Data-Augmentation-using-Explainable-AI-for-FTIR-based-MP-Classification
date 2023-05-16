from utils import  utils
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential,Model
from keras.layers import LSTM, Dense, Activation,RepeatVector,Input
from keras.utils.vis_utils import plot_model
def normalizedWavelength(wavenumbers):
    if wavenumbers[0] > wavenumbers[-1]:
        rng = wavenumbers[0] - wavenumbers[-1]
    else:
        rng = wavenumbers[-1] - wavenumbers[0]
    half_rng = rng / 2
    normalized_wns = (wavenumbers - np.mean(wavenumbers)) / half_rng
    return normalized_wns
#
polymerName, waveLength, intensity, polymerID, x_each, y_each = utils.parseData11('data/D4_4_publication11.csv', 2, 1763)
#
# import numpy as np
# import matplotlib.pyplot as plt
# from keras.models import Sequential
# waveLength=np.array(waveLength,dtype=float)
# waveLength=normalizedWavelength(waveLength)
# # np.random.seed(0)
# # points = 500
# waveLength=waveLength.reshape(1,1761,1)
# waveLength=waveLength.reshape(1,1761,1)
# print(intensity[0].shape)
# intre=intensity[0].reshape(1761,1)
#
# model = Sequential()
# # model.add(Dense(1024, activation='relu', input_dim=1))
# # model.add(Dense(512, activation='relu'))
# # model.add(Dense(256, activation='relu'))
# # model.add(Dense(256, activation='relu'))
# # model.add(Dense(256, activation='relu'))
# # model.add(Dense(512, activation='relu'))
# # model.add(Dense(1054, activation='sigmoid'))
# # model.add(Dense(1))
#
# # model.compile(loss='mse', optimizer="adam")
# inputs = Input(shape=(1,1))
# encoded = LSTM(1024)(inputs)
# decoded = RepeatVector(1)(encoded)
# decoded = LSTM(1, return_sequences=True,activation='sigmoid')(decoded)
# model = Model(inputs, decoded)
#
# # model.add(LSTM(input_dim=1, output_dim=6, return_sequences=True))
# # model.add(LSTM(100, return_sequences=False))
# # model.add(Dense(output_dim=1))
# # model.add(Activation('linear'))
# # model.summary()
#
# model.compile(loss='mse', optimizer='rmsprop')
# model.fit(waveLength, intre, epochs=50)
#
# predictions = model.predict(waveLength)
# plt.scatter(waveLength, intensity[0])
# plt.plot(waveLength, predictions, 'ro')
# plt.show()
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
def LSTMPolynomiaRegression(intensity,n):
    seq = 10
    augmentation=[]
    ylabel=[]
    for item in intensity:
        y=item
        train = np.array(y).astype(float)
        train=train.reshape(-1,1)
        # scaler = MinMaxScaler()
        # train = scaler.fit_transform(train)
        data = []
        for i in range(len(train) - seq - 1):
             data.append(train[i: i + seq + 1])
        data = np.array(data).astype('float64')

        x = data[:, :-1]
        y = data[:, -1]
        split1= int(data.shape[0] * 0.9)
        split2 = int(data.shape[0] * 0.1)
        xremain=[]
        for i in range(11):
            xremain.append(item[i])

        # print(xremain.shape)
        train_x = x[1:split1]
        train_x = np.concatenate((train_x,x[split2:]),axis=0)
        train_y = y[1:split1]
        train_y=np.concatenate((train_y, y[split2:]), axis=0)

        test_x = x #[split:]
        test_y = y #[split:]

        train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
        test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

        model = Sequential()
        model.add(LSTM(units=1, return_sequences=True))
        model.add(LSTM(20, return_sequences=False))
        model.add(Dense(1))
        model.add(Activation('linear'))


        model.compile(loss='mse', optimizer='adam')

        model.fit(train_x, train_y, batch_size=1024, epochs=2)
        model.summary()
        predict_y = model.predict(test_x)
        predict_y = np.reshape(predict_y, (predict_y.size,))
        print(predict_y.shape)
        #predict_y=np.concatenate((xremain,predict_y),axis=1)
        for i in range(len(predict_y)):
            xremain.append(predict_y[i])
        print(xremain)
        xremain=np.array(xremain,dtype=float)
        # plt.plot(waveLength,item,'r')
        # plt.plot(waveLength,xremain,'b')
        # plt.show()
        print('xremain',xremain.shape)
        augmentation.append(xremain)
        ylabel.append(n)


    return augmentation,ylabel

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 实例化 Pipeline
# waveLength=np.array(waveLength)
# waveLength=normalizedWavelength(waveLength)
# waveLength=waveLength.reshape(-1,1)
# poly=PolynomialFeatures(degree=2)
# poly_X=poly.fit_transform(waveLength)
# print(poly_X)
# lr=LinearRegression()
# lr.fit(poly_X,intensity[0])
#
#
# x_test=np.linspace(waveLength.max(), waveLength.min(), 1000)
# x_test=x_test.reshape(-1,1)
# x_test_poly=poly.fit_transform(x_test)
# y_predict = lr.predict(x_test_poly)
# plt.scatter(waveLength, intensity[0])
# plt.plot(x_test, y_predict[np.argsort(x_test)], color='r')
# plt.show()
# predict_y = scaler.inverse_transform([[i] for i in predict_y])
# test_y = scaler.inverse_transform(test_y)
# fig2 = plt.figure(2)
# plt.plot(predict_y, 'g')
# plt.plot(test_y, 'r')
# plt.show()
# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)