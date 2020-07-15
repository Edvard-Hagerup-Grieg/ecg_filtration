import json
import os
import pickle as pkl
import sys

import BaselineWanderRemoval as bwr
import numpy as np

DATA_PATH = "D:\\data\\"
DATA_FILENAME = "data_2033.json"
DIAG_FILENAME = "diagnosis.json"
PKL_FILENAME = "data_2033.pkl"
HOLTER_PATH = "D:\\data\\holters\\"
HOLTER_FILENAME = "holter"
MIT_PATH = "D:\\data\\mit\\mit_dataset.pkl"

LEADS_NAMES = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
FREQUENCY_OF_DATASET = 500


def parser(path):
    try:
        infile = open(path + DATA_FILENAME, 'rb')
        data = json.load(infile)
        diag_dict = get_diag_dict()

        X = []
        Y = []
        for id in data.keys():

            leads = data[id]['Leads']
            diagnosis = data[id]['StructuredDiagnosisDoc']

            y = []
            try:
                for diag in diag_dict.keys():
                    y.append(diagnosis[diag])
            except KeyError:
                print("\nThe patient " + id + " is not included in the final dataset. Reason: no diagnosis.")
                continue
            y = np.where(y, 1, 0)

            x = []
            try:
                for lead in LEADS_NAMES:
                    rate = int(leads[lead]['SampleRate'] / FREQUENCY_OF_DATASET)
                    x.append(leads[lead]['Signal'][::rate])
            except KeyError:
                print("\nThe patient " + id + " is not included in the final dataset. Reason: no lead.")
                continue

            X.append(x)
            Y.append(y)

        X = np.array(X)
        Y = np.array(Y)
        X = np.swapaxes(X, 1, 2)

        print("The dataset is parsed.")
        print("X shape: ", X.shape)
        print("Y shape: ", Y.shape)

        return {"x": X, "y": Y}

    except FileNotFoundError:
        print("File " + DATA_FILENAME + " has not found.\nThe specified folder (" + path +
              ") must contain files with data (" + DATA_FILENAME +
              ") and file with structure of diagnosis (" + DIAG_FILENAME + ").")
        sys.exit(0)


def get_diag_dict():
    def deep(data, diag_list):
        for diag in data:
            if diag['type'] == 'diagnosis':
                diag_list.append(diag['name'])
            else:
                deep(diag['value'], diag_list)

    try:
        infile = open(DATA_PATH + DIAG_FILENAME, 'rb')
        data = json.load(infile)

        diag_list = []
        deep(data, diag_list)

        diag_num = list(range(len(diag_list)))
        diag_dict = dict(zip(diag_list, diag_num))

        return diag_dict

    except FileNotFoundError:
        print("File " + DIAG_FILENAME + " has not found.\nThe specified folder (" + DATA_PATH +
              ") must contain files with data (" + DATA_FILENAME +
              ") and file with structure of diagnosis (" + DIAG_FILENAME + ").")
        sys.exit(0)


def load_dataset(folder_path=DATA_PATH):
    if not os.path.exists(folder_path + PKL_FILENAME):
        xy = parser(folder_path)
        fix_bw(xy, folder_path)

    with open(folder_path + PKL_FILENAME, 'rb') as infile:
        dataset = pkl.load(infile)

    return dataset


def load_holter(patient = 0, folder_path=HOLTER_PATH):
    with open(folder_path + HOLTER_FILENAME + str(patient) + ".pkl", 'rb') as infile:
        dataset = pkl.load(infile)
    return dataset


def load_mit():
    with open(MIT_PATH, 'rb') as infile:
        dataset = pkl.load(infile)[:,:,:1]
    return dataset


def fix_bw(xy, folder_path):
    print("Baseline wondering fixing is started. It's take some time.")

    X = xy["x"]
    patients_num = X.shape[0]
    for i in range(patients_num):
        print("\rSignal %s/" % str(i + 1) + str(patients_num) + ' is fixed.', end='')
        for j in range(X.shape[2]):
            X[i, :, j] = bwr.fix_baseline_wander(X[i, :, j], FREQUENCY_OF_DATASET)
    xy['x'] = X

    with open(folder_path + PKL_FILENAME, 'rb') as outfile:
        pkl.dump(xy, outfile)

    print("The dataset is saved.")


def normalize_data(X):
    mn = X.mean(axis=0)
    st = X.std(axis=0)
    x_std = np.zeros(X.shape)
    for i in range(X.shape[0]):
        x_std[i] = (X[i] - mn) / st
    return x_std


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = load_dataset()['x']

    for i in range(x.shape[0]):
        plt.plot(x[i, :, 0])
        plt.show()