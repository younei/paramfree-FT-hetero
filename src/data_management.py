
import numpy as np
import math
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from scipy import io as sio
import pickle
import copy 


class Settings:

    def __init__(self, dictionary, struct_name=None):
        if struct_name is None:
            self.__dict__.update(**dictionary)
        else:
            temp_settings = Settings(dictionary)
            setattr(self, struct_name, temp_settings)

    def add_settings(self, dictionary, struct_name=None):

        if struct_name is None:
            self.__dict__.update(dictionary)
        else:
            if hasattr(self, struct_name):
                temp_settings = getattr(self, struct_name)
                temp_settings.__dict__.update(dictionary)
            else:
                temp_settings = Settings(dictionary)
            setattr(self, struct_name, temp_settings)


class DataHandler:

    def __init__(self, settings):
        settings.add_settings({'n_all_tasks': settings.data.n_tr_tasks + settings.data.n_val_tasks + settings.data.n_test_tasks}, 'data')
        self.settings = settings
        self.features_tr = [None] * settings.data.n_all_tasks
        self.features_ts = [None] * settings.data.n_all_tasks
        self.labels_tr = [None] * settings.data.n_all_tasks
        self.labels_ts = [None] * settings.data.n_all_tasks
        self.oracle_unconditional = None
        self.oracle_conditional = None

        self.tr_task_indexes = None
        self.val_task_indexes = None  
        self.test_task_indexes = None   

        if self.settings.data.dataset == 'synthetic-regression':
            self.synthetic_regression_data_gen()
        elif self.settings.data.dataset == 'synthetic-regression-multi-clusters':
            self.synthetic_regression_data_MULTI_CLUSTERS_gen()
        elif self.settings.data.dataset == 'synthetic-regression-multi-clusters-BIS':
            self.synthetic_regression_data_MULTI_CLUSTERS_BIS_gen()
        elif self.settings.data.dataset == 'circle':
            self.synthetic_circle_data_gen()
        elif self.settings.data.dataset == 'schools':
            self.schools_data_gen()
        elif self.settings.data.dataset == 'lenk':
            self.lenk_data_gen()
        else:
            raise ValueError('Invalid dataset')

    # synthetic data 1 cluster
    def synthetic_regression_data_gen(self):

        self.oracle_unconditional = 4 * np.ones(self.settings.data.n_dims) 

        for task_idx in range(self.settings.data.n_all_tasks):

            # generating and normalizing the inputs
            features = np.random.randn(self.settings.data.n_all_points, self.settings.data.n_dims)
            features = features / norm(features, axis=1, keepdims=True)
            features = features + 1   
            

            # generating the weight vectors
            weight_vector = self.oracle_unconditional + np.random.normal(loc=np.zeros(self.settings.data.n_dims),
                                                                         scale=1).ravel()

            # generating labels and adding noise
            clean_labels = features @ weight_vector

            signal_to_noise_ratio = 1
            standard_noise = np.random.randn(self.settings.data.n_all_points)
            noise_std = np.sqrt(np.var(clean_labels) / (signal_to_noise_ratio * np.var(standard_noise)))
            noisy_labels = clean_labels + noise_std * standard_noise

            # split into training and test
            tr_indexes, ts_indexes = train_test_split(np.arange(0, self.settings.data.n_all_points),
                                                      test_size=self.settings.data.ts_points_pct)
            features_tr = features[tr_indexes]
            labels_tr = noisy_labels[tr_indexes]

            features_ts = features[ts_indexes]
            labels_ts = noisy_labels[ts_indexes]

            self.features_tr[task_idx] = features_tr
            self.features_ts[task_idx] = features_ts
            self.labels_tr[task_idx] = labels_tr
            self.labels_ts[task_idx] = labels_ts

        self.tr_task_indexes = np.arange(0, self.settings.data.n_tr_tasks)
        self.val_task_indexes = np.arange(self.settings.data.n_tr_tasks,
                                          self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks)
        self.test_task_indexes = np.arange(self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks,
                                           self.settings.data.n_all_tasks)

        # add side information 
        self.all_side_info = copy.deepcopy(self.features_tr) 

    # synthetic data MULTI clusters w_\rho = 4
    def synthetic_regression_data_MULTI_CLUSTERS_gen(self):

        number_clusters = 2
        n_tasks = self.settings.data.n_all_tasks
        first_centroid_weights = 8 * np.ones(self.settings.data.n_dims)
        second_centroid_weights = np.zeros(self.settings.data.n_dims)
        all_centroids_weights = [first_centroid_weights, second_centroid_weights]
        c_inputs = 1  # constant to control || first_centroid_inputs - second_centroid_inputs ||
        first_centroid_inputs = c_inputs * np.ones(self.settings.data.n_dims)
        second_centroid_inputs = - first_centroid_inputs
        all_centroids_inputs = [first_centroid_inputs, second_centroid_inputs]
        clusters_belonging_indexes = np.random.randint(number_clusters, size=(1, n_tasks))
        clusters_belonging_indexes = clusters_belonging_indexes

        self.oracle_unconditional = np.mean(all_centroids_weights, axis=0)

        for task_idx in range(self.settings.data.n_all_tasks):
            cluster_idx = clusters_belonging_indexes[0, task_idx]
            centroid_inputs = all_centroids_inputs[cluster_idx]
            centroid_weights = all_centroids_weights[cluster_idx]

            # generating and normalising the inputs
            features = np.random.randn(self.settings.data.n_all_points, self.settings.data.n_dims)
            features = features / norm(features, axis=1, keepdims=True)
            features = centroid_inputs + features
            
            

            # generating the weight vectors
            weight_vector = centroid_weights + np.random.normal(loc=np.zeros(self.settings.data.n_dims),
                                                                scale=1).ravel()

            # generating labels and adding noise
            clean_labels = features @ weight_vector

            signal_to_noise_ratio = 1
            standard_noise = np.random.randn(self.settings.data.n_all_points)
            noise_std = np.sqrt(np.var(clean_labels) / (signal_to_noise_ratio * np.var(standard_noise)))
            noisy_labels = clean_labels + noise_std * standard_noise

            # split into training and test
            tr_indexes, ts_indexes = train_test_split(np.arange(0, self.settings.data.n_all_points),
                                                      test_size=self.settings.data.ts_points_pct)
            features_tr = features[tr_indexes]
            labels_tr = noisy_labels[tr_indexes]
            features_ts = features[ts_indexes]
            labels_ts = noisy_labels[ts_indexes]

            self.features_tr[task_idx] = features_tr
            self.features_ts[task_idx] = features_ts
            self.labels_tr[task_idx] = labels_tr
            self.labels_ts[task_idx] = labels_ts

        self.tr_task_indexes = np.arange(0, self.settings.data.n_tr_tasks)
        self.val_task_indexes = np.arange(self.settings.data.n_tr_tasks,
                                          self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks)
        self.test_task_indexes = np.arange(self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks,
                                           self.settings.data.n_all_tasks)

        # add side information 
        self.all_side_info = copy.deepcopy(self.features_tr) 

    # synthetic data MULTI clusters w_\rho = 0
    def synthetic_regression_data_MULTI_CLUSTERS_BIS_gen(self):

        number_clusters = 2
        n_tasks = self.settings.data.n_all_tasks
        c_weights = 4  # constant to control || first_centroid_weights - second_centroid_weights ||
        first_centroid_weights = c_weights * np.ones(self.settings.data.n_dims)
        second_centroid_weights = - first_centroid_weights
        all_centroids_weights = [first_centroid_weights, second_centroid_weights]
        c_inputs = 1  # constant to control || first_centroid_inputs - second_centroid_inputs ||
        first_centroid_inputs = c_inputs * np.ones(self.settings.data.n_dims)
        second_centroid_inputs = - first_centroid_inputs
        all_centroids_inputs = [first_centroid_inputs, second_centroid_inputs]
        clusters_belonging_indexes = np.random.randint(number_clusters, size=(1, n_tasks))
        #clusters_belonging_indexes = clusters_belonging_indexes

        self.oracle_unconditional = np.mean(all_centroids_weights, axis=0)

        for task_idx in range(self.settings.data.n_all_tasks):

            cluster_idx = clusters_belonging_indexes[0, task_idx]
            centroid_inputs = all_centroids_inputs[cluster_idx]
            centroid_weights = all_centroids_weights[cluster_idx]

            # generating the inputs
            features = np.random.randn(self.settings.data.n_all_points, self.settings.data.n_dims)
            features = features / norm(features, axis=1, keepdims=True)
            features = centroid_inputs + features 

            # generating the weight vectors
            weight_vector = centroid_weights + np.random.normal(loc=np.zeros(self.settings.data.n_dims),
                                                                scale=1).ravel()

            # generating labels and adding noise
            clean_labels = features @ weight_vector

            signal_to_noise_ratio = 1
            standard_noise = np.random.randn(self.settings.data.n_all_points)
            noise_std = np.sqrt(np.var(clean_labels) / (signal_to_noise_ratio * np.var(standard_noise)))
            noisy_labels = clean_labels + noise_std * standard_noise

            # split into training and test
            tr_indexes, ts_indexes = train_test_split(np.arange(0, self.settings.data.n_all_points),
                                                      test_size=self.settings.data.ts_points_pct)
            features_tr = features[tr_indexes]
            labels_tr = noisy_labels[tr_indexes]
            features_ts = features[ts_indexes]
            labels_ts = noisy_labels[ts_indexes]

            self.features_tr[task_idx] = features_tr
            self.features_ts[task_idx] = features_ts
            self.labels_tr[task_idx] = labels_tr
            self.labels_ts[task_idx] = labels_ts

        self.tr_task_indexes = np.arange(0, self.settings.data.n_tr_tasks)
        self.val_task_indexes = np.arange(self.settings.data.n_tr_tasks,
                                          self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks)
        self.test_task_indexes = np.arange(self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks,
                                           self.settings.data.n_all_tasks)

        # add side information 
        self.all_side_info = copy.deepcopy(self.features_tr) 

    # synthetic data - circle
    def synthetic_circle_data_gen(self):

        c_weights = 0
        circle_center = c_weights * np.ones(self.settings.data.n_dims)
        c_inputs = 1
        inputs_center = c_inputs * np.ones(self.settings.data.n_dims)
        self.all_side_info = np.random.uniform(low=0, high=1, size=self.settings.data.n_all_tasks)

        self.oracle_unconditional = circle_center

        for task_idx in range(self.settings.data.n_all_tasks):

            s = self.all_side_info[task_idx]
            mean_w = np.zeros(self.settings.data.n_dims)
            mean_w[0] = self.settings.data.radius_w * math.cos(2 * s * math.pi)
            mean_w[1] = self.settings.data.radius_w * math.sin(2 * s * math.pi)

            # generating and normalizing the inputs
            features = np.random.randn(self.settings.data.n_all_points, self.settings.data.n_dims)
            features = features / norm(features, axis=1, keepdims=True)
            features = features + inputs_center

            # generating the weight vectors
            weight_vector = mean_w + self.settings.data.sigma_w * np.random.normal(loc=np.zeros(self.settings.data.n_dims),
                                                                                   scale=1).ravel()

            # generating labels and adding noise
            clean_labels = features @ weight_vector

            signal_to_noise_ratio = 1
            standard_noise = np.random.randn(self.settings.data.n_all_points)
            noise_std = np.sqrt(np.var(clean_labels) / (signal_to_noise_ratio * np.var(standard_noise)))
            noisy_labels = clean_labels + noise_std * standard_noise

            # split into training and test
            tr_indexes, ts_indexes = train_test_split(np.arange(0, self.settings.data.n_all_points),
                                                      test_size=self.settings.data.ts_points_pct)
            features_tr = features[tr_indexes]
            labels_tr = noisy_labels[tr_indexes]

            features_ts = features[ts_indexes]
            labels_ts = noisy_labels[ts_indexes]

            self.features_tr[task_idx] = features_tr
            self.features_ts[task_idx] = features_ts
            self.labels_tr[task_idx] = labels_tr
            self.labels_ts[task_idx] = labels_ts

        self.tr_task_indexes = np.arange(0, self.settings.data.n_tr_tasks)
        self.val_task_indexes = np.arange(self.settings.data.n_tr_tasks,
                                          self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks)
        self.test_task_indexes = np.arange(self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks,
                                           self.settings.data.n_all_tasks)

    # Schools dataset
    def schools_data_gen(self):

        
        temp = sio.loadmat('/Users/youning/Desktop/DISSERTATION/Project/DISSER_CODE/data/schoolData.mat')
        all_features = [temp['X'][0][i].T for i in range(len(temp['X'][0]))]
        all_labels = temp['Y'][0]
        n_tasks = len(all_features)

        shuffled_tasks = list(range(n_tasks))
        np.random.shuffle(shuffled_tasks)

        for task_idx, task in enumerate(shuffled_tasks):

            # normalizing the inputs
            features = all_features[task]
            features = features / norm(features, axis=1, keepdims=True)

            labels = all_labels[task].ravel()
            n_points = len(labels) 

            # split into training and test
            tr_indexes, ts_indexes = train_test_split(np.arange(0, n_points), test_size=self.settings.data.ts_points_pct)
            features_tr = features[tr_indexes]
            labels_tr = labels[tr_indexes]
            features_ts = features[ts_indexes]
            labels_ts = labels[ts_indexes]

            self.features_tr[task_idx] = features_tr
            self.features_ts[task_idx] = features_ts
            self.labels_tr[task_idx] = labels_tr
            self.labels_ts[task_idx] = labels_ts

        self.tr_task_indexes = np.arange(0, self.settings.data.n_tr_tasks)
        self.val_task_indexes = np.arange(self.settings.data.n_tr_tasks, self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks)
        self.test_task_indexes = np.arange(self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks, self.settings.data.n_all_tasks)
        # add side information 
        self.all_side_info = copy.deepcopy(self.features_tr) 


    # Lenk dataset
    def lenk_data_gen(self):

        temp_test = sio.loadmat('/Users/youning/Desktop/DISSERTATION/Project/DISSER_CODE/data/lenk_te_triple.mat')
        temp_train = sio.loadmat('/Users/youning/Desktop/DISSERTATION/Project/DISSER_CODE/data/lenk_tr_triple.mat')
        train_data = temp_train['Traindata']  # 2880x15  last feature output (score from 0 to 10) (144 tasks of 20 elements)
        test_data = temp_test['Testdata']  # 720x15 last feature is y (score from 0 to 10) (26 tasks of 20 elements)

        Y = train_data[:, 14]
        Y_test = test_data[:, 14]
        X = train_data[:, :14]
        X_test = test_data[:, :14]

        # n_tasks = 180  # --> n_tot_tasks
        n_tasks = 540
        n_tot = 20
        ne_tr = 16  # number of elements on train set per task
        ne_test = 4  # number of elements on test set per task


        def split_tasks(data, nt, number_of_elements):
            return [data[i * number_of_elements:(i + 1) * number_of_elements] for i in range(nt)]

        data_m = split_tasks(X, n_tasks, ne_tr)
        labels_m = split_tasks(Y, n_tasks, ne_tr)

        data_test_m = split_tasks(X_test, n_tasks, ne_test)
        labels_test_m = split_tasks(Y_test, n_tasks, ne_test)

        shuffled_tasks = list(range(self.settings.data.n_all_tasks))
        np.random.shuffle(shuffled_tasks)

        for task_idx, task in enumerate(shuffled_tasks):

            es = np.random.permutation(len(labels_m[task]))
            es = list(range(len(labels_m[task])))

            X_train, Y_train = data_m[task][es], labels_m[task][es]
            X_test, Y_test = data_test_m[task], labels_test_m[task]

            Y_train = Y_train.ravel()
            X_train = X_train



            self.features_tr[task_idx] = X_train / norm(X_train, axis=1, keepdims=True)
            self.features_ts[task_idx] = X_test / norm(X_test, axis=1, keepdims=True)
            self.labels_tr[task_idx] = Y_train   
            self.labels_ts[task_idx] = Y_test     

        self.tr_task_indexes = np.arange(0, self.settings.data.n_tr_tasks)
        self.val_task_indexes = np.arange(self.settings.data.n_tr_tasks,
                                          self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks)
        self.test_task_indexes = np.arange(self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks,
                                           self.settings.data.n_all_tasks)

        # add side information 
        self.all_side_info = copy.deepcopy(self.features_tr)  