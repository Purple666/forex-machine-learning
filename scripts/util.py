import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models import Model


def create_model(params):

    return Model(params)


def load_all_data():

    data = pd.read_pickle('../data/EURUSD-2010_2018-closeAsk.pkl')

    return data


def split_intervals(data_intervals, h_split):

    output_train = []
    output_label = []

    interval_length = np.shape(data_intervals[0])[0]
    split_idx = int(h_split*interval_length)

    for j, data_interval in enumerate(data_intervals):
        output_train.append(data_interval[:split_idx])
        output_label.append(data_interval[split_idx:])

    return output_train, output_label


def get_performance(cluster_indexes, labels):

    clusters = np.sort(np.unique(cluster_indexes))

    output_performance = []
    output_labels = []

    for c in clusters:
        indxs = np.where(cluster_indexes == c)[0]
        labels_ = np.asarray(labels)[indxs]
        output_labels.append(labels_)
        output_performance.append(np.sum(labels_))

    return output_performance, output_labels


def plot_hist(data, hists, title):

    if hists:
        fig = plt.figure()
        plt.hist(data, bins=30)
        plt.grid()
        plt.title(title)


def get_labels(label_intervals):

    labels = []

    for data_ in label_intervals:

        labels.append(data_[-1]-data_[0])

    return labels


def set_system_parameters():
    parser = argparse.ArgumentParser(description='Unsupervised learning approach to Forex trading analysis')
    parser.add_argument('--date_init', nargs='?', help='backtesting initial date', default='2010-01-01')
    parser.add_argument('--date_end', nargs='?', help='backtesting end date', default='2018-01-01')
    parser.add_argument('--h_init', nargs='?', help='time interval intial hour', default=9)
    parser.add_argument('--h_end', nargs='?', help='time interval final hour', default=17)
    parser.add_argument('--h_split', nargs='?', help='interval splitting parameter', default=0.8)
    parser.add_argument('--n_clusters', nargs='?', help='number of clusters in clustering algorithm', default=10)
    parser.add_argument('--hists', nargs='?', help='boolean plot performance distribution histogram', default="F")
    parser.add_argument('--action_th', nargs='?', help='', default="0.55")

    args = parser.parse_args()

    date_init = args.date_init
    date_end = args.date_end
    h_init = int(args.h_init)
    h_end = int(args.h_end)
    h_split = float(args.h_split)
    n_clusters = int(args.n_clusters)
    hists = args.hists in ["T", "t", "true", "True", "1"]
    action_th = float(args.action_th)

    params = {'date_init': date_init,
              'date_end': date_end,
              'h_init': h_init,
              'h_end': h_end,
              'h_split': h_split,
              'n_clusters': n_clusters,
              'hists': hists,
              'action_th': action_th}

    return params



