import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


class Model:

    def __init__(self, params):
        n_clusters = params['n_clusters']
        self.clustering_algo = KMeans(n_clusters=n_clusters)
        self.params = params
        self.trading_actions = None

    def fit(self, data):
        # intervals
        groups = self.group_data_intervals_by_similarity(data)
        # _groups_ is a list with length n_clusters

        properties = self.analyze_properties(groups)

        self.trading_actions = self.groups_to_trading_actions(properties)

    def transform(self, val_data):
        # TODO: implement
        pass

    def group_data_intervals_by_similarity(self, data):
        params = self.params

        h_init = params['h_init']
        h_end = params['h_end']
        date_init = params['date_init']
        date_end = params['date_end']
        h_split = params['h_split']
        n_clusters = params['n_clusters']

        time_interval = [h_init, h_end]

        intervals = self.get_intervals(date_init, date_end, time_interval)

        # data to intervals

        data_intervals = self.data_to_intervals(data, intervals)

        train_data_intervals = data_intervals[:, :int(h_split * np.shape(data_intervals)[1])]

        # normalize train data

        normalized_train_intervals = self.normalize_intervals(train_data_intervals)

        # cluster data

        self.fit_clustering_algorithm(normalized_train_intervals, n_clusters)

        cluster_indexes = self.get_cluster_indexes(self)

        groups = []

        for cluster in range(n_clusters):
            indexes = np.where(cluster_indexes == cluster)[0]
            groups.append(data_intervals[indexes])

        return groups

    def data_to_intervals(self, data, intervals):
        # TODO: relocate method

        temp = np.zeros((len(intervals), len(intervals[0])))

        counter = 0
        for j, interval in enumerate(intervals):

            data_ = data[interval]

            if not np.isnan(data_).any():
                temp[counter, :] = data_
                counter += 1

        output = temp[:counter, :]

        return output

    def normalize_intervals(self, train_intervals):
        # TODO: relocate method

        return normalize(train_intervals, axis=1)

    def fit_clustering_algorithm(self, normalized_train_intervals):

        self.clustering_algo.fit(normalized_train_intervals)

    def get_cluster_indexes(self):

        return self.clustering_algo.labels_

    def analyze_properties(self, groups):

        h_split = self.params['h_split']

        properties = []
        for cluster_indx, data in enumerate(groups):

            interval_final = data[:, int(np.shape(data)[1] * h_split):]

            deltas = []
            cluster_size = np.shape(data)[0]
            for j in range(cluster_size):
                deltas.append(interval_final[j][-1] - interval_final[j][0])

            deltas = np.asarray(deltas)

            properties.append({'positive_return': sum(deltas[np.where(deltas > 0)[0]]) / sum(abs(deltas)),
                               'cluster_size': cluster_size,
                               'cluster_indx': cluster_indx})

        # TODO: visualize properties

        return properties

    def groups_to_trading_actions(self, properties):

        th = self.params['action_th']

        trading_actions = {'trading_actions': {}}  # TODO: what is the most adequate structure for this object ?
        for j, group_properties in enumerate(properties):
            if group_properties['positive_return'] > th:
                action = 0  # map numbers to actions
            elif group_properties['positive_return'] < 1 - th:
                action = 1  # map numbers to actions
            else:
                action = 2  # map numbers to actionss

            trading_actions['trading_actions'][j] = ({'cluster': str(group_properties['cluster_indx']),
                                                      'action': action})

        return trading_actions