import numpy as np
from src.general_functions import unit_ball_projection, subgradient, loss


class BasicBias:
    def __init__(self, fixed_bias):
        self.fixed_bias = fixed_bias
        self.step_size_range = [10**i for i in np.linspace(-1, 4, 16)]
        self.w = None

    def fit(self, data, task_indexes):

        performance = []
        for task_idx, task in enumerate(getattr(data, task_indexes)):
            x = data.features_tr[task]
            y = data.labels_tr[task]
            n_points, n_dims = x.shape

            best_perf = np.Inf
            for step_idx, step_size in enumerate(self.step_size_range):
                curr_untranslated_weights = np.zeros(n_dims)
                curr_weights = curr_untranslated_weights + self.fixed_bias
                all_weight_vectors = []
                all_losses = []
                shuffled_indexes = list(range(n_points))
                np.random.shuffle(shuffled_indexes)
                for iteration, curr_point_idx in enumerate(shuffled_indexes):
                    prev_untranslated_weights = curr_untranslated_weights
                    prev_weights = curr_weights

                    # receive a new datapoint
                    curr_x = x[curr_point_idx, :]
                    curr_y = y[curr_point_idx]

                    # compute the gradient
                    subgrad = subgradient(curr_x, curr_y, prev_weights, loss_name='absolute')
                    full_gradient = subgrad * curr_x

                    # update weight vector
                    curr_untranslated_weights = prev_untranslated_weights - step_size * full_gradient
                    curr_weights = curr_untranslated_weights + self.fixed_bias
                    all_weight_vectors.append(curr_weights)

                    if len(all_weight_vectors) < 2:
                        final_w = curr_weights
                    else:
                        final_w = np.mean(all_weight_vectors, axis=0)
                    loss_thing = loss(x, y, final_w, loss_name='absolute')
                    all_losses.append(loss_thing)

                curr_perf = loss(data.features_ts[task], data.labels_ts[task], final_w, loss_name='absolute')
                if curr_perf < best_perf:
                    best_perf = curr_perf
                    best_step = step_size
            performance.append(best_perf)
            print(performance)

### NOTE: SELF.W is not used here  

class ParameterFreeFixedBiasVariation:
    def __init__(self, fixed_bias, L = 1, R = 1):
        self.L = L
        self.R = R
        self.fixed_bias = fixed_bias
        self.w = None
        self.magnitude_betting_fraction = 0
        self.magnitude_wealth = 1

    def fit(self, data):

        all_mtl_performances = []
        all_errors = []
        total_points = 0
        for task_idx, task in enumerate(data.tr_task_indexes):
            x = data.features_tr[task] ## (num_tr, dim)
            y = data.labels_tr[task]  ## (numtr,)
            n_points, n_dims = x.shape 
            total_points = total_points + n_points 

            wealth_range = [1] ## this is 'e' ; no loop below 
            for idx, wealth in enumerate(wealth_range):
                curr_bet_fraction = self.magnitude_betting_fraction # b_i

                curr_wealth = wealth   # self.magnitude_wealth u_i = e
                curr_magnitude = curr_bet_fraction * curr_wealth # p_1 = b_1 * u_1
                curr_direction = np.random.randn(n_dims) # v_1 : randomly generated 

                all_weight_vectors = [] # store all weights 
                all_h = [] 

                shuffled_indexes = list(range(n_points)) # list of i = [1,2,...,n]
                np.random.shuffle(shuffled_indexes) # shuffled_indexes is shuffled i 
                for iteration, curr_point_idx in enumerate(shuffled_indexes):
                    iteration = iteration + 1 ## start from 1 (cuz 0 in Python by default)
                    prev_direction = curr_direction # v
                    prev_bet_fraction = curr_bet_fraction # b
                    prev_wealth = curr_wealth # u
                    prev_magnitude = curr_magnitude # p=bu

                    # update weight vector
                    weight_vector = prev_magnitude * prev_direction + self.fixed_bias
                    all_weight_vectors.append(weight_vector)

                    # receive a new datapoint
                    curr_x = x[curr_point_idx, :] # extract the feature array from the batch 
                    curr_y = y[curr_point_idx] # extract corresponding label from the batch 

                    all_errors.append(loss(curr_x, curr_y, weight_vector, loss_name='absolute')) # loss incured by initial predictor 

                    # compute the gradient - to update the next 
                    subgrad = subgradient(curr_x, curr_y, weight_vector, loss_name='absolute')
                    full_gradient = subgrad * curr_x

                    # define step size
                    step_size = 1 / (self.L * self.R) * np.sqrt(2 / iteration)

                    # update direction
                    curr_direction = unit_ball_projection(prev_direction - step_size * full_gradient)

                    # update magnitude_wealth
                    curr_wealth = prev_wealth - (1 / (self.R * self.L)) * full_gradient @ prev_direction * prev_magnitude

                    ## Sophisticated Version of Coin-Betting 
                    # h_thing
                    h = (1 / (self.R * self.L)) * full_gradient @ prev_direction * (1 / (1 - (1 / (self.R * self.L)) * (full_gradient @ prev_direction) * prev_bet_fraction))
                    all_h.append(h)
                    a_thing = 1 + np.sum([curr_h**2 for curr_h in all_h])

                    # update magnitude_betting_fraction
                    curr_bet_fraction = np.max([np.min([prev_bet_fraction - (2 / (2 - np.log(3))) * (h / a_thing), 1/2]), -1/2])

                    # update magnitude
                    curr_magnitude = curr_bet_fraction * curr_wealth

                    if len(all_weight_vectors) < 2:
                        final_w = weight_vector # if only updated one time - exclude the initial weight 
                    else:
                        final_w = np.mean(all_weight_vectors, axis=0)

                ## END OF ONE TASK - INCUR LOSS AT THE END OF THE TASK 
                curr_test_perf = loss(data.features_ts[task], data.labels_ts[task], final_w, loss_name='absolute')

                all_mtl_performances.append(curr_test_perf) 
        
        ## return 1: (average) across-tasks error  \sum_{t=1}^T avg l_t() 
        ## return 2:
        ## all_errors : computed on training data 
        return (task_idx + 1) * [np.nanmean(all_mtl_performances)], total_points * [np.nanmean(all_errors)]

##
## return cummulative error for ITL, test error for ITL 