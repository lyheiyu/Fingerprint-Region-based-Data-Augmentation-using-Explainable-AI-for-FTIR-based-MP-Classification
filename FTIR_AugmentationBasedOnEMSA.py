import numpy as np

import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
import pandas as pd
from typing import Union as U, Tuple as T
from utils import utils
import numpy as np
from sklearn.preprocessing import StandardScaler
import random

def emsc(spectra: np.ndarray, wavenumbers: np.ndarray, order: int = 2,
         reference: np.ndarray = None,
         constituents: np.ndarray = None,
         return_coefs: bool = False) -> U[np.ndarray, T[np.ndarray, np.ndarray]]:
    """
    Preprocess all spectra with EMSC
    :param spectra: ndarray of shape [n_samples, n_channels]
    :param wavenumbers: ndarray of shape [n_channels]
    :param order: order of polynomial
    :param reference: reference spectrum
    :param constituents: ndarray of shape [n_consituents, n_channels]
    Except constituents it can also take orthogonal vectors,
    for example from PCA.
    :param return_coefs: if True returns coefficients
    [n_samples, n_coeffs], where n_coeffs = 1 + len(costituents) + (order + 1).
    Order of returned coefficients:
    1) b*reference +                                    # reference coeff
    k) c_0*constituent[0] + ... + c_k*constituent[k] +  # constituents coeffs
    a_0 + a_1*w + a_2*w^2 + ...                         # polynomial coeffs
    :return: preprocessed spectra
    """
    if reference is None:
        reference = np.mean(spectra, axis=0)
    print(spectra)
    print(reference)
    reference = reference[:, np.newaxis]

    # squeeze wavenumbers to approx. range [-1; 1]
    # use if else to support uint types
    if wavenumbers[0] > wavenumbers[-1]:
        rng = wavenumbers[0] - wavenumbers[-1]
    else:
        rng = wavenumbers[-1] - wavenumbers[0]
    half_rng = rng / 2
    normalized_wns = (wavenumbers - np.mean(wavenumbers)) / half_rng

    polynomial_columns = [np.ones(len(wavenumbers))]
    for j in range(1, order + 1):
        polynomial_columns.append(normalized_wns ** j)
    polynomial_columns = np.stack(polynomial_columns).T

    # spectrum = X*coefs + residues
    # least squares -> A = (X.T*X)^-1 * X.T; coefs = A * spectrum
    if constituents is None:
        columns = (reference, polynomial_columns)
    else:
        columns = (reference, constituents.T, polynomial_columns)

    X = np.concatenate(columns, axis=1)
    A = np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T)

    spectra_columns = spectra.T
    coefs = np.dot(A, spectra_columns)
    residues = spectra_columns - np.dot(X, coefs)

    preprocessed_spectra = (reference + residues/coefs[0]).T

    if return_coefs:
        return preprocessed_spectra, coefs.T

    return preprocessed_spectra

class EMSA:
    """
    Extended Multiplicative Signal Augmentation
    Generates balanced batches of augmentated spectra
    """

    def __init__(self, std_of_params, wavenumbers, reference, order=2):
        """
        :param std_of_params: array of length (order+2), which
        :param reference: reference spectrum that was used in EMSC model
        :param order: order of emsc
        contains the std for each coefficient
        """
        self.order = order
        self.std_of_params = std_of_params
        self.ref = reference
        self.X = None
        self.A = None
        self.__create_x_and_a(wavenumbers)

    def generator(self, spectra, labels,
                  equalize_subsampling=False, shuffle=True,
                  batch_size=32):
        """ generates batches of transformed spectra"""
        spectra = np.asarray(spectra)
        labels = np.asarray(labels)

        if self.std_of_params is None:
            coefs = np.dot(self.A, spectra.T)
            self.std_of_params = coefs.std(axis=1)

        if equalize_subsampling:
            indexes = self.__rearrange_spectra(labels)
        else:
            indexes = np.arange(len(spectra))

        cur = 0
        while True:
            if shuffle:
                si = indexes[np.random.randint(len(indexes),
                                               size=batch_size)]
            else:
                si = indexes.take(range(cur, cur + batch_size),
                                  mode='wrap')
                cur += batch_size

            yield self.__batch_transform(spectra[si]), labels[si]

    def __rearrange_spectra(self, labels):
        """ returns indexes of data rearranged in the way of 'balance'"""
        classes = np.unique(labels, axis=0)

        if len(labels.shape) == 2:
            grouped = [np.where(np.all(labels == l, axis=1))[0]
                       for l in classes]
        else:
            grouped = [np.where(labels == l)[0] for l in classes]
        iters_cnt = max([len(g) for g in grouped])

        indexes = []
        for i in range(iters_cnt):
            for g in grouped:
                # take cyclic sample from group
                indexes.append(np.take(g, i, mode='wrap'))

        return np.array(indexes)

    def __create_x_and_a(self, wavenumbers):
        """
        Builds X matrix from spectra in such way that columns go as
        reference w^0 w^1 w^2 ... w^n, what corresponds to coefficients
        b, a, d, e, ...
        and caches the solution self.A = (X^T*X)^(-1)*X^T
        :param spectra:
        :param wavenumbers:
        :return: nothing, but creates two self.X and self.A
        """
        # squeeze wavenumbers to approx. range [-1; 1]
        # use if else to support uint types
        if wavenumbers[0] > wavenumbers[-1]:
            rng = wavenumbers[0] - wavenumbers[-1]
        else:
            rng = wavenumbers[-1] - wavenumbers[0]
        half_rng = rng / 2
        normalized_wns = (wavenumbers - np.mean(wavenumbers)) / half_rng

        self.polynomial_columns = [np.ones_like(wavenumbers)]
        for j in range(1, self.order + 1):
            self.polynomial_columns.append(normalized_wns ** j)

        self.X = np.stack((self.ref, *self.polynomial_columns), axis=1)
        self.A = np.dot(np.linalg.pinv(np.dot(self.X.T, self.X)), self.X.T)

    def __batch_transform(self, spectra):
        spectra_columns = spectra.T

        # b, a, d, e, ...

        coefs = np.dot(self.A, spectra_columns)
        residues = spectra_columns - np.dot(self.X, coefs)

        new_coefs = coefs.copy()

        # wiggle coefficients
        for i in range(len(coefs)):
            new_coefs[i] += np.random.normal(0,
                                             self.std_of_params[i],
                                             len(spectra))

        # Fix if multiplication parameter sampled negative
        mask = new_coefs[0] <= 0
        if np.any(mask):
            # resample multiplication parameter to be positive
            n_resamples = mask.sum()
            new_coefs[0][mask] = np.random.uniform(0, coefs[0][mask],
                                                   n_resamples)


        return (np.dot(self.X, new_coefs) + residues * new_coefs[0] / coefs[0]).T

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    polymerName, waveLength, intensity, polymerID, x_each, y_each = utils.parseData11('D4_4_publication11.csv', 1200, 1700)


    cmTotal = np.zeros((11, 11))
    m = 0
    t_report = []
    scoreTotal = np.zeros(5)
    for seedNum in range(20):
        x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.7, random_state=seedNum)
        waveLength = np.array(waveLength, dtype=np.float)
        datas=[]
        datas2 = []
        PN = utils.getPN('D4_4_publication11.csv')
        for n in range(len(PN)):
            numSynth = 2
            indicesPS = [l for l, id in enumerate(y_train) if id == n]
            intensityForLoop = x_train[indicesPS]
            datas.append(intensityForLoop)
            datas2.append(intensityForLoop)
        for itr in range(0,11):
            _, coefs_ = emsc(
                datas[itr], waveLength,reference=None,
                order=2,
                return_coefs=True)

            coefs_std = coefs_.std(axis=0)
            indicesPS = [l for l, id in enumerate(y_train) if id == itr]
            label=y_train[indicesPS]


            reference=datas[itr].mean(axis=0)
            emsa = EMSA(coefs_std, waveLength, reference, order=2)

            generator = emsa.generator(datas[itr], label,
                equalize_subsampling=False, shuffle=False,
                batch_size=100)


            augmentedSpectrum = []
            for i, batch in enumerate(generator):
                if i >2:
                    break
                augmented = []
                for augmented_spectrum, label in zip(*batch):

                    plt.plot(waveLength, augmented_spectrum, label=label)
                    augmented.append (augmented_spectrum)
                augmentedSpectrum.append(augmented)
                # plt.gca().invert_xaxis()
                # plt.legend()
                # plt.show()
            augmentedSpectrum=np.array(augmentedSpectrum)
            y_add=[]
            for item in augmentedSpectrum[0]:
                y_add.append(itr)
            from sklearn.preprocessing import normalize
            augmentedSpectrum[0] = normalize(augmentedSpectrum[0], 'max')
            x_train=np.concatenate((x_train,augmentedSpectrum[0]),axis=0)
            y_train=np.concatenate((y_train,y_add),axis=0)

        # randnum = random.randint(0, 100)
        # random.seed(randnum)
        # random.shuffle(x_train)
        # random.seed(randnum)
        # random.shuffle(y_train)
        #x_train, x_test0, y_train, y_test0 = train_test_split(x_train, y_train, test_size=0.1, random_state=1)
        model = svm.SVC(C=0.3, kernel='linear', decision_function_shape='ovo')
        model = model.fit(x_train, y_train)
        y_pre = model.predict(x_test)

        utils.printScore(y_test, y_pre)
        PN = utils.getPN('D4_4_publication11.csv')
        t = classification_report(y_test, y_pre, target_names=PN, output_dict=True)
        # SVM_report = pd.DataFrame(t)
        # SVM_report.to_csv('SVM_report5.csv')
        cm = confusion_matrix(y_test, y_pre)
        #utils.plot_confusion_matrix(cm,PN,'EMSA_SVM')
        scores = utils.printScore(y_test, y_pre)

        cmTotal = cmTotal + cm
        scoreTotal += scores
        m += 1
        # ax[0].plot(x, y1, '-k')
        # modelMLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128, 128), random_state=1)
        # modelMLP.fit(x_train,y_train)
        # y_pre = modelMLP.predict(x_test)
        #
        # utils.printScore(y_test, y_pre)
        # PN = utils.getPN('D4_4_publication11.csv')
        # t = classification_report(y_test, y_pre, target_names=PN, output_dict=True)
        # cm = confusion_matrix(y_test, y_pre)
        #utils.plot_confusion_matrix(cm, PN, 'EMSA_MLP')
        print(m)
    print(scoreTotal / m)
    # maxnumber.append(sum(scoreTotal / m) )
    #
    cmTotal = cmTotal / m

    utils.plot_confusion_matrix(cmTotal, PN, 'ESMA data_SVM')
    fig, ax = plt.subplots(nrows=3, ncols=1)
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 30,
             }
    for item in _:
        ax[0].plot(waveLength, item, '-r')
    for item in datas[0]:
        ax[1].plot(waveLength, item, '-r')
    for item in augmentedSpectrum[0]:
        ax[2].plot(waveLength, item, '-r')
    labels0 = ax[0].get_xticklabels() + ax[0].get_yticklabels()
    labels1 = ax[1].get_xticklabels() + ax[1].get_yticklabels()
    labels2 = ax[2].get_xticklabels() + ax[2].get_yticklabels()
    ax[0].tick_params(labelsize=15)
    ax[1].tick_params(labelsize=15)
    ax[2].tick_params(labelsize=15)
    # [label0.set_fontname('normal') for label0 in labels0]
    # [label0.set_fontstyle('normal') for label0 in labels0]
    ax[0].set_title('EMSC', font2)
    ax[1].set_title('Original', font2)
    ax[2].set_title('EMSA', font2)
    plt.show()