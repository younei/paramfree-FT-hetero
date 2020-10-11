import numpy as np
import pandas as pd
from src.general_functions import unit_ball_projection, subgradient, loss


class ParameterFreeLazyClassic:
    def __init__(self, L=1, R=1):
        self.L = L
        self.R = R
        self.all_weight_vectors = None
        self.all_meta_parameters = None
        self.meta_magnitude_betting_fraction = 0
        self.meta_magnitude_wealth = 1
        self.inner_magnitude_betting_fraction = 0
        self.inner_magnitude_wealth = 1

    def fit(self, data):
        curr_meta_magnitude_betting_fraction = self.meta_magnitude_betting_fraction
        curr_meta_magnitude_wealth = self.meta_magnitude_wealth
        curr_meta_magnitude = curr_meta_magnitude_betting_fraction * curr_meta_magnitude_wealth
        curr_meta_direction = np.zeros(data.features_tr[0].shape[1])

        all_individual_cum_errors = []
        best_mtl_performances = []

        total_iter = 0
        all_meta_parameters = []
        all_final_weight_vectors = []
        for task_iteration, task in enumerate(data.tr_task_indexes):
            x = data.features_tr[task]
            y = data.labels_tr[task]

            task_iteration = task_iteration + 1
            prev_meta_direction = curr_meta_direction
            prev_meta_magnitude_betting_fraction = curr_meta_magnitude_betting_fraction
            prev_meta_magnitude_wealth = curr_meta_magnitude_wealth
            prev_meta_magnitude = curr_meta_magnitude

            # update meta-parameter
            meta_parameter = prev_meta_magnitude * prev_meta_direction
            all_meta_parameters.append(meta_parameter)

            # initialize the inner parameters
            n_points, n_dims = x.shape
            curr_inner_magnitude_betting_fraction = self.inner_magnitude_betting_fraction
            curr_inner_magnitude_wealth = self.inner_magnitude_wealth
            curr_inner_magnitude = curr_inner_magnitude_betting_fraction * curr_inner_magnitude_wealth
            curr_inner_direction = np.zeros(x.shape[1])

            temp_weight_vectors = []
            all_gradients = []
            shuffled_indexes = list(range(n_points))
            # np.random.shuffle(shuffled_indexes)
            for inner_iteration, curr_point_idx in enumerate(shuffled_indexes):
                inner_iteration = inner_iteration + 1
                prev_inner_direction = curr_inner_direction
                prev_inner_magnitude_betting_fraction = curr_inner_magnitude_betting_fraction
                prev_inner_magnitude_wealth = curr_inner_magnitude_wealth
                prev_inner_magnitude = curr_inner_magnitude

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
                all_gradients.append(full_gradient)

                # define inner step size
                inner_step_size = (1 / (self.L * self.R)) * np.sqrt(2 / inner_iteration)

                # update inner direction
                curr_inner_direction = unit_ball_projection(prev_inner_direction - inner_step_size * full_gradient)

                # update inner magnitude_wealth
                curr_inner_magnitude_wealth = prev_inner_magnitude_wealth - 1 / (self.R * self.L) * full_gradient @ prev_inner_direction * prev_inner_magnitude

                # update magnitude_betting_fraction
                curr_inner_magnitude_betting_fraction = (1/inner_iteration) * ((inner_iteration-1) * prev_inner_magnitude_betting_fraction - (1/(self.L*self.R))*(full_gradient @ prev_inner_direction))

                # update magnitude
                curr_inner_magnitude = curr_inner_magnitude_betting_fraction * curr_inner_magnitude_wealth

            # define total iteration
            total_iter = total_iter + n_points

            # compute meta-gradient
            meta_gradient = np.sum(all_gradients, axis=0)

            # define meta step size
            meta_step_size = (1 / (self.L * self.R * n_points)) * np.sqrt(2 / task_iteration)

            # update meta-direction
            curr_meta_direction = unit_ball_projection(prev_meta_direction - meta_step_size * meta_gradient)

            # update meta-magnitude_wealth
            curr_meta_magnitude_wealth = prev_meta_magnitude_wealth - (1 / (self.R * self.L * n_points)) * meta_gradient @ prev_meta_direction * prev_meta_magnitude

            # update meta-magnitude_betting_fraction
            curr_meta_magnitude_betting_fraction = (1/task_iteration) * ((task_iteration-1) * prev_meta_magnitude_betting_fraction - (1 / (self.L * self.R * n_points)) * (meta_gradient @ prev_meta_direction))

            # update meta-magnitude
            curr_meta_magnitude = curr_meta_magnitude_betting_fraction * curr_meta_magnitude_wealth

            all_final_weight_vectors.append(np.mean(temp_weight_vectors, axis=0))
            all_test_errors = []
            for idx, curr_test_task in enumerate(data.tr_task_indexes[:task_iteration]):
                all_test_errors.append(loss(data.features_ts[curr_test_task], data.labels_ts[curr_test_task], all_final_weight_vectors[idx], loss_name='absolute'))
            best_mtl_performances.append(np.nanmean(all_test_errors))

        self.all_meta_parameters = all_meta_parameters
        return best_mtl_performances, pd.DataFrame(all_individual_cum_errors).rolling(window=10 ** 10, min_periods=1).mean().values.ravel()


class ParameterFreeLazyVariation:
    def __init__(self, L=1, R=1):
        self.L = L
        self.R = R
        self.all_weight_vectors = None
        self.all_meta_parameters = None
        self.meta_magnitude_betting_fraction = 0
        self.meta_magnitude_wealth = 1
        self.inner_magnitude_betting_fraction = 0
        self.inner_magnitude_wealth = 1

    def fit(self, data):
        curr_meta_magnitude_betting_fraction = self.meta_magnitude_betting_fraction
        curr_meta_magnitude_wealth = self.meta_magnitude_wealth
        curr_meta_magnitude = curr_meta_magnitude_betting_fraction * curr_meta_magnitude_wealth
        curr_meta_direction = np.zeros(data.features_tr[0].shape[1])

        all_h_meta = []
        all_individual_cum_errors = []
        best_mtl_performances = []

        total_iter = 0
        all_meta_parameters = []
        all_final_weight_vectors = []
        for task_iteration, task in enumerate(data.tr_task_indexes):
            x = data.features_tr[task]
            y = data.labels_tr[task]

            task_iteration = task_iteration + 1
            prev_meta_direction = curr_meta_direction
            prev_meta_magnitude_betting_fraction = curr_meta_magnitude_betting_fraction
            prev_meta_magnitude_wealth = curr_meta_magnitude_wealth
            prev_meta_magnitude = curr_meta_magnitude

            # update meta-parameter
            meta_parameter = prev_meta_magnitude * prev_meta_direction
            all_meta_parameters.append(meta_parameter)

            # initialize the inner parameters
            n_points, n_dims = x.shape
            curr_inner_magnitude_betting_fraction = self.inner_magnitude_betting_fraction
            curr_inner_magnitude_wealth = self.inner_magnitude_wealth
            curr_inner_magnitude = curr_inner_magnitude_betting_fraction * curr_inner_magnitude_wealth
            curr_inner_direction = np.zeros(x.shape[1])

            all_h_inner = []

            temp_weight_vectors = []
            all_gradients = []
            shuffled_indexes = list(range(n_points))
            # np.random.shuffle(shuffled_indexes)
            for inner_iteration, curr_point_idx in enumerate(shuffled_indexes):
                inner_iteration = inner_iteration + 1
                prev_inner_direction = curr_inner_direction
                prev_inner_magnitude_betting_fraction = curr_inner_magnitude_betting_fraction
                prev_inner_magnitude_wealth = curr_inner_magnitude_wealth
                prev_inner_magnitude = curr_inner_magnitude

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
                all_gradients.append(full_gradient)

                # define inner step size
                inner_step_size = (1 / (self.L * self.R)) * np.sqrt(2 / inner_iteration)

                # update inner direction
                curr_inner_direction = unit_ball_projection(prev_inner_direction - inner_step_size * full_gradient)

                # update inner magnitude_wealth
                curr_inner_magnitude_wealth = prev_inner_magnitude_wealth - 1 / (self.R * self.L) * full_gradient @ prev_inner_direction * prev_inner_magnitude

                # update magnitude_betting_fraction
                h_inner = (1 / (self.R * self.L)) * full_gradient @ prev_inner_direction * (1 / (1 - (1 / (self.R * self.L)) * full_gradient @ prev_inner_direction * prev_inner_magnitude_betting_fraction))
                all_h_inner.append(h_inner)
                a_thing_inner = 1 + np.sum([curr_h ** 2 for curr_h in all_h_inner])

                curr_inner_magnitude_betting_fraction = np.max([np.min([prev_inner_magnitude_betting_fraction - (2 / (2 - np.log(3))) * (h_inner / a_thing_inner), 1 / 2]), -1 / 2])

                # update magnitude
                curr_inner_magnitude = curr_inner_magnitude_betting_fraction * curr_inner_magnitude_wealth

            # define total iteration
            total_iter = total_iter + n_points

            # compute meta-gradient
            meta_gradient = np.sum(all_gradients, axis=0)

            # define meta step size
            meta_step_size = (1 / (self.L * self.R * n_points)) * np.sqrt(2 / task_iteration)

            # update meta-direction
            curr_meta_direction = unit_ball_projection(prev_meta_direction - meta_step_size * meta_gradient)

            # update meta-magnitude_wealth
            curr_meta_magnitude_wealth = prev_meta_magnitude_wealth - (1 / (self.R * self.L * n_points)) * meta_gradient @ prev_meta_direction * prev_meta_magnitude

            # update meta-magnitude_betting_fraction
            h_meta = (1 / (self.R * self.L * n_points)) * (meta_gradient @ prev_meta_direction) * (1 / (1 - (1 / (self.R * self.L * n_points)) * (meta_gradient @ prev_meta_direction) * prev_meta_magnitude_betting_fraction))
            all_h_meta.append(h_meta)
            a_thing_meta = 1 + np.sum([curr_h ** 2 for curr_h in all_h_meta])

            curr_meta_magnitude_betting_fraction = np.max([np.min([prev_meta_magnitude_betting_fraction - (2 / (2 - np.log(3))) * (h_meta / a_thing_meta), 1/2]), -1/2])

            # update meta-magnitude
            curr_meta_magnitude = curr_meta_magnitude_betting_fraction * curr_meta_magnitude_wealth

            all_final_weight_vectors.append(np.mean(temp_weight_vectors, axis=0))
            all_test_errors = []
            for idx, curr_test_task in enumerate(data.tr_task_indexes[:task_iteration]):
                all_test_errors.append(loss(data.features_ts[curr_test_task], data.labels_ts[curr_test_task], all_final_weight_vectors[idx], loss_name='absolute'))
            best_mtl_performances.append(np.nanmean(all_test_errors))


        self.all_meta_parameters = all_meta_parameters
        return best_mtl_performances, pd.DataFrame(all_individual_cum_errors).rolling(window=10 ** 10, min_periods=1).mean().values.ravel()
