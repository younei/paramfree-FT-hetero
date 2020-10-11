import numpy as np
import pandas as pd
from src.general_functions import unit_ball_projection, subgradient, loss


class ParameterFreeAggressiveClassic:
    def __init__(self, L=1,  R=1):
        self.L = L
        self.R = R

    def fit(self, data):
        meta_wealth_range = [1]
        inner_wealth_range = [1]

        best_cumsum_perf = np.Inf
        for _, meta_wealth in enumerate(meta_wealth_range):
            for _, inner_wealth in enumerate(inner_wealth_range):
                all_individual_cum_errors = []

                curr_meta_fraction = 0
                curr_meta_wealth = meta_wealth
                curr_meta_magnitude = curr_meta_fraction * curr_meta_wealth
                curr_meta_direction = np.zeros(data.features_tr[0].shape[1])

                all_final_weight_vectors = []

                all_mtl_performances = []
                all_meta_parameters = []
                for task_iteration, task in enumerate(data.tr_task_indexes):
                    x = data.features_tr[task]
                    y = data.labels_tr[task]

                    task_iteration = task_iteration + 1

                    # initialize the inner parameters
                    n_points, n_dims = x.shape
                    curr_inner_fraction = 0
                    curr_inner_wealth = inner_wealth
                    curr_inner_magnitude = curr_inner_fraction * curr_inner_wealth
                    curr_inner_direction = np.zeros(n_dims)

                    temp_weight_vectors = []
                    shuffled_indexes = list(range(n_points))
                    # np.random.shuffle(shuffled_indexes)
                    for inner_iteration, curr_point_idx in enumerate(shuffled_indexes):
                        inner_iteration = inner_iteration + 1

                        prev_meta_direction = curr_meta_direction
                        prev_meta_fraction = curr_meta_fraction
                        prev_meta_wealth = curr_meta_wealth
                        prev_meta_magnitude = curr_meta_magnitude

                        prev_inner_direction = curr_inner_direction
                        prev_inner_fraction = curr_inner_fraction
                        prev_inner_wealth = curr_inner_wealth
                        prev_inner_magnitude = curr_inner_magnitude

                        # define total iteration
                        total_iter = self.__general_iteration(task_iteration, n_points, inner_iteration)

                        # update meta-parameter
                        meta_parameter = prev_meta_magnitude * prev_meta_direction
                        all_meta_parameters.append(meta_parameter)

                        # update inner weight vector
                        weight_vector = prev_inner_magnitude * prev_inner_direction + meta_parameter
                        temp_weight_vectors.append(weight_vector)

                        # receive a new datapoint
                        curr_x = x[curr_point_idx, :]
                        curr_y = y[curr_point_idx]

                        all_individual_cum_errors.append(loss(curr_x, curr_y, weight_vector, loss_name='absolute'))

                        # compute the gradient
                        subgrad = subgradient(curr_x, curr_y, weight_vector, loss_name='absolute')
                        full_gradient = subgrad * curr_x

                        # define meta step size
                        meta_step_size = (1 / (self.L * self.R)) * np.sqrt(2 / total_iter)

                        # update meta-direction
                        curr_meta_direction = unit_ball_projection(prev_meta_direction - meta_step_size * full_gradient)

                        # define inner step size
                        inner_step_size = (1 / (self.L * self.R)) * np.sqrt(2 / inner_iteration)

                        # update inner direction
                        curr_inner_direction = unit_ball_projection(prev_inner_direction - inner_step_size * full_gradient)

                        # update meta-magnitude_wealth
                        curr_meta_wealth = prev_meta_wealth - (1 / (self.R * self.L)) * full_gradient @ prev_meta_direction * prev_meta_magnitude

                        # update meta-magnitude_betting_fraction
                        curr_meta_fraction = (1/total_iter) * ((total_iter-1) * prev_meta_fraction - (1/(self.L*self.R))*(full_gradient @ prev_meta_direction))

                        # update meta-magnitude
                        curr_meta_magnitude = curr_meta_fraction * curr_meta_wealth

                        # update inner magnitude_wealth
                        curr_inner_wealth = prev_inner_wealth - 1 / (self.R * self.L) * full_gradient @ prev_inner_direction * prev_inner_magnitude

                        # update magnitude_betting_fraction
                        curr_inner_fraction = (1 / inner_iteration) * ((inner_iteration - 1) * prev_inner_fraction - (1 / (self.L * self.R)) * (full_gradient @ prev_inner_direction))

                        # update magnitude
                        curr_inner_magnitude = curr_inner_fraction * curr_inner_wealth

                    all_final_weight_vectors.append(np.mean(temp_weight_vectors, axis=0))
                    all_test_errors = []
                    for idx, curr_test_task in enumerate(data.tr_task_indexes[:task_iteration]):
                        all_test_errors.append(loss(data.features_ts[curr_test_task], data.labels_ts[curr_test_task], all_final_weight_vectors[idx], loss_name='absolute'))
                    all_mtl_performances.append(np.nanmean(all_test_errors))

                if np.nanmean(all_individual_cum_errors) < best_cumsum_perf:
                    best_cumsum_perf = np.nanmean(all_individual_cum_errors)
                    best_cumsum_performances = all_individual_cum_errors
                    best_mtl_performances = all_mtl_performances
                    print('inner wealth: %8.2f | meta wealth: %8.2f |       perf: %10.3f' % (inner_wealth, meta_wealth, np.nanmean(all_individual_cum_errors)))
                else:
                    print('inner wealth: %8.2f | meta wealth: %8.2f | perf: %10.3f' % (inner_wealth, meta_wealth, np.nanmean(all_individual_cum_errors)))
        return best_mtl_performances, pd.DataFrame(best_cumsum_performances).rolling(window=10 ** 10, min_periods=1).mean().values.ravel()

    @staticmethod
    def __general_iteration(t, n, i):
        return (t - 1) * n + i


class ParameterFreeAggressiveVariation:
    def __init__(self, L=1, R=1):
        self.L = L
        self.R = R
        self.all_weight_vectors = []
        self.all_meta_parameters = []
        self.meta_magnitude_betting_fraction = 0
        self.inner_magnitude_betting_fraction = 0

    def fit(self, data):
        meta_wealth_range = [1]
        inner_wealth_range = [1]

        for meta_idx, meta_wealth in enumerate(meta_wealth_range):
            for inner_idx, inner_wealth in enumerate(inner_wealth_range):
                all_individual_cum_errors = []

                curr_meta_fraction = self.meta_magnitude_betting_fraction
                curr_meta_wealth = meta_wealth
                curr_meta_magnitude = curr_meta_fraction * curr_meta_wealth
                curr_meta_direction = np.zeros(data.features_tr[0].shape[1])

                all_final_weight_vectors = []
                all_h_meta = []
                all_mtl_performances = []
                all_meta_parameters = []
                for task_iteration, task in enumerate(data.tr_task_indexes):
                    x = data.features_tr[task]
                    y = data.labels_tr[task]

                    task_iteration = task_iteration + 1

                    # initialize the inner parameters
                    n_points, n_dims = x.shape
                    curr_inner_fraction = self.inner_magnitude_betting_fraction
                    curr_inner_wealth = inner_wealth
                    curr_inner_magnitude = curr_inner_fraction * curr_inner_wealth
                    curr_inner_direction = np.zeros(n_dims)

                    all_h_inner = []

                    temp_weight_vectors = []
                    shuffled_indexes = list(range(n_points))
                    # np.random.shuffle(shuffled_indexes)
                    for inner_iteration, curr_point_idx in enumerate(shuffled_indexes):
                        inner_iteration = inner_iteration + 1

                        prev_meta_direction = curr_meta_direction
                        prev_meta_fraction = curr_meta_fraction
                        prev_meta_wealth = curr_meta_wealth
                        prev_meta_magnitude = curr_meta_magnitude

                        prev_inner_direction = curr_inner_direction
                        prev_inner_fraction = curr_inner_fraction
                        prev_inner_wealth = curr_inner_wealth
                        prev_inner_magnitude = curr_inner_magnitude

                        # define total iteration
                        total_iter = self.__general_iteration(task_iteration, n_points, inner_iteration)

                        # update meta-parameter
                        meta_parameter = prev_meta_magnitude * prev_meta_direction
                        all_meta_parameters.append(meta_parameter)

                        # update inner weight vector
                        weight_vector = prev_inner_magnitude * prev_inner_direction + meta_parameter
                        temp_weight_vectors.append(weight_vector)

                        # receive a new datapoint
                        curr_x = x[curr_point_idx, :]
                        curr_y = y[curr_point_idx]

                        all_individual_cum_errors.append(loss(curr_x, curr_y, weight_vector, loss_name='absolute'))

                        # compute the gradient
                        subgrad = subgradient(curr_x, curr_y, weight_vector, loss_name='absolute')
                        full_gradient = subgrad * curr_x

                        # define meta step size
                        meta_step_size = (1 / (self.L * self.R)) * np.sqrt(2 / total_iter)

                        # update meta-direction
                        curr_meta_direction = unit_ball_projection(prev_meta_direction - meta_step_size * full_gradient)

                        # define inner step size
                        inner_step_size = (1 / (self.L * self.R)) * np.sqrt(2 / inner_iteration)

                        # update inner direction
                        curr_inner_direction = unit_ball_projection(prev_inner_direction - inner_step_size * full_gradient)

                        # update meta-magnitude_wealth
                        curr_meta_wealth = prev_meta_wealth - (1 / (self.R * self.L)) * full_gradient @ prev_meta_direction * prev_meta_magnitude

                        # meta h thing
                        h_meta = (1 / (self.R * self.L)) * (full_gradient @ prev_meta_direction) * (1 / (1 - (1 / (self.R * self.L)) * (full_gradient @ prev_meta_direction) * prev_meta_fraction))
                        all_h_meta.append(h_meta)
                        a_thing_meta = 1 + np.sum([curr_h ** 2 for curr_h in all_h_meta])

                        # update meta-magnitude_betting_fraction
                        curr_meta_fraction = np.max([np.min([prev_meta_fraction - (2 / (2 - np.log(3))) * (h_meta / a_thing_meta), 1 / 2]), -1 / 2])

                        # update meta-magnitude
                        curr_meta_magnitude = curr_meta_fraction * curr_meta_wealth

                        # update inner magnitude_wealth
                        curr_inner_wealth = prev_inner_wealth - 1 / (self.R * self.L) * full_gradient @ prev_inner_direction * prev_inner_magnitude

                        # update magnitude_betting_fraction
                        h_inner = (1 / (self.R * self.L)) * full_gradient @ prev_inner_direction * (1 / (1 - (1 / (self.R * self.L)) * full_gradient @ prev_inner_direction * prev_inner_fraction))
                        all_h_inner.append(h_inner)
                        a_thing_inner = 1 + np.sum([curr_h**2 for curr_h in all_h_inner])
                        # update magnitude_betting_fraction
                        curr_inner_fraction = np.max([np.min([prev_inner_fraction - (2 / (2 - np.log(3))) * (h_inner / a_thing_inner), 1/2]), -1/2])

                        # update magnitude
                        curr_inner_magnitude = curr_inner_fraction * curr_inner_wealth

                    all_final_weight_vectors.append(np.mean(temp_weight_vectors, axis=0))
                    all_test_errors = []
                    for idx, curr_test_task in enumerate(data.tr_task_indexes[:task_iteration]):
                        all_test_errors.append(loss(data.features_ts[curr_test_task], data.labels_ts[curr_test_task], all_final_weight_vectors[idx], loss_name='absolute'))
                    all_mtl_performances.append(np.nanmean(all_test_errors))

        return all_mtl_performances, pd.DataFrame(all_individual_cum_errors).rolling(window=10 ** 10, min_periods=1).mean().values.ravel()

    @staticmethod
    def __general_iteration(t, n, i):
        return (t - 1) * n + i
