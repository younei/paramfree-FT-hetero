import numpy as np
import pandas as pd
from src.general_functions import loss, subgradient, feature_map, unit_ball_projection, Frobenius

class ParameterFreeConditionalLazyFineTuning:
    def __init__(self, L=1, R=1, K=1, loss_name='absolute', feature_map_name=None, r=None, W=None):
        
        ## Assumption Value 
        self.L = L
        self.R = R 
        self.K = K
        ## Feature map parameter 
        self.feature_map_name = feature_map_name
        self.r = r 
        self.W = W
        # Wealth 
        self.meta_M_wealth = 1
        self.meta_b_wealth = 1
        self.inner_wealth = 1
        # Betting fractionsss
        self.meta_M_betting_fraction = 0
        self.meta_b_betting_fraction = 0
        self.inner_magnitude_betting_fraction = 0  
        # Loss name 
        self.loss_name = loss_name    
        # Estimated parameter storage 
        self.all_final_weight_vectors = []
        self.all_meta_parameters = {"M":[],
                                    "b":[],
                                    "theta":[]}

    def fit(self, data):
        
        # Feature shape
        test_for_shape = feature_map(data.all_side_info[0], data.labels_tr[0], self.feature_map_name, self.r, self.W)
		# Initialize meta-parameter M 
        curr_meta_M_betting_fraction = self.meta_M_betting_fraction  # B 
        curr_meta_M_wealth = self.meta_M_wealth # U = E
        curr_meta_M_magnitude = curr_meta_M_betting_fraction * curr_meta_M_wealth # P = BU
        curr_meta_M_direction = np.zeros([data.features_tr[0].shape[1], test_for_shape.shape[0]])

        # Initialize meta-parameter b
        curr_meta_b_betting_fraction = self.meta_b_betting_fraction # B 
        curr_meta_b_wealth = self.meta_b_wealth # U = E 
        curr_meta_b_magnitude = curr_meta_b_betting_fraction * curr_meta_b_wealth # P = BU
        curr_meta_b_direction = np.zeros(data.features_tr[0].shape[1]) # V 
        
        all_individual_cum_errors = []
        best_mtl_performances = []


        # For t = 1,..., T 
        for task_iteration, task in enumerate(data.tr_task_indexes, 1):
            # Retrive Dataset Z_t = (x_t, y_t)
            x = data.features_tr[task]
            y = data.labels_tr[task]
            # Get side information s_t associated to each task 
            s = data.all_side_info[task] 
            # Get transformed feature Φ(s_t) 
            x_trasf_feature = feature_map(s, y, self.feature_map_name, self.r, self.W)


            # Meta M  
            prev_meta_M_direction = curr_meta_M_direction
            prev_meta_M_betting_fraction = curr_meta_M_betting_fraction
            prev_meta_M_wealth = curr_meta_M_wealth
            prev_meta_M_magnitude = curr_meta_M_magnitude

            # Update meta-parameter transition matrix M 
            meta_M = prev_meta_M_magnitude * prev_meta_M_direction
            self.all_meta_parameters["M"].append(meta_M)

            # Meta b  
            prev_meta_b_direction = curr_meta_b_direction
            prev_meta_b_betting_fraction = curr_meta_b_betting_fraction
            prev_meta_b_wealth = curr_meta_b_wealth
            prev_meta_b_magnitude = curr_meta_b_magnitude

            # Update meta-parameter shift b 
            meta_b = prev_meta_b_magnitude * prev_meta_b_direction 
            self.all_meta_parameters["b"].append(meta_b)

            # Update meta parameter theta 
            meta_parameter = meta_M @ x_trasf_feature + meta_b 
            self.all_meta_parameters["theta"].append(meta_parameter)
            
            # Initialize inner parameter 
            n_points, n_dims = x.shape # n, d
            curr_inner_betting_fraction = self.inner_magnitude_betting_fraction # b
            curr_inner_wealth = self.inner_wealth # e
            curr_inner_magnitude = curr_inner_betting_fraction * curr_inner_wealth # p = bu 
            curr_inner_direction = np.zeros(n_dims) # v

            temp_weight_vectors = [] # to store weight vector obtained within-task
            all_gradient = [] # to store g_{t,i} 
            shuffled_indexes = list(range(n_points))
            # np.random.shuffle(shuffled_indexes) 

            # FOR i = 1, ... , n
            for inner_iteration, curr_point_idx in enumerate(shuffled_indexes, 1):

                # inner  
                prev_inner_direction = curr_inner_direction
                prev_inner_betting_fraction = curr_inner_betting_fraction
                prev_inner_wealth = curr_inner_wealth
                prev_inner_magnitude = curr_inner_magnitude

                # update inner weight vector 
                weight_vector = prev_inner_magnitude * prev_inner_direction + meta_parameter
                # store inner weight vector 
                temp_weight_vectors.append(weight_vector)

                # receive a new data point z_{t,i} = (x_{t,i}, y_{t,i})
                curr_x = x[curr_point_idx, :]
                curr_y = y[curr_point_idx]
                # incur error 
                all_individual_cum_errors.append(loss(curr_x, curr_y, weight_vector, loss_name=self.loss_name))

                # compute the gradient 
                subgrad = subgradient(curr_x, curr_y, weight_vector, loss_name=self.loss_name)
                # g_{t,i}
                full_gradient = subgrad * curr_x
                all_gradient.append(full_gradient)

                ##------------------##
                ##   Inner Update   ##
                ##------------------##

                # define inner step size 
                inner_step_size = (1 / (self.L * self.R)) * np.sqrt(2 / inner_iteration)
                # update direction 
                curr_inner_direction = unit_ball_projection(prev_inner_direction - inner_step_size * full_gradient)
                # update wealth 
                curr_inner_wealth = prev_inner_wealth - 1 / (self.R * self.L) * full_gradient @ prev_inner_direction * prev_inner_magnitude
                # update betting fraction 
                curr_inner_betting_fraction = (1/inner_iteration) * ((inner_iteration - 1) * prev_inner_betting_fraction - (1/(self.L * self.R)) * (full_gradient @ prev_inner_direction))
                # update magnitude
                curr_inner_magnitude = curr_inner_betting_fraction * curr_inner_wealth

            # compute meta graident G_t = sum g_{t,i} 
            meta_gradient = np.sum(all_gradient, axis=0)

            # receive G_t \Phi(s_t)
            meta_gradient_map = meta_gradient[:,None] @ x_trasf_feature[:,None].T 

            ##------------------##
            ##     M Update     ##
            ##------------------##
            # define meta step size for M 
            M_step_size = (1/(self.L * self.K * self.R * n_points)) *  np.sqrt(2 / task_iteration)
            # update direction 
            curr_meta_M_direction = unit_ball_projection(prev_meta_M_direction - M_step_size * meta_gradient_map)
            # update wealth 
            curr_meta_M_wealth = prev_meta_M_wealth - (1 / (self.R * self.L * self.K * n_points)) * Frobenius(meta_gradient_map, prev_meta_M_direction) * prev_meta_M_magnitude
            # update betting fraction 
            curr_meta_M_betting_fraction = (1/task_iteration) * ((task_iteration-1) * prev_meta_M_betting_fraction - (1/(self.L*self.R*self.K*n_points)) * Frobenius(meta_gradient_map, prev_meta_M_direction))
            # update magnitude 
            curr_meta_M_magnitude = curr_meta_M_betting_fraction * curr_meta_M_wealth

            ##------------------##
            ##     b Update     ##
            ##------------------##
            # define meta step size for b 
            b_step_size = (1 / (self.L * self.R * n_points)) *  np.sqrt(2 / task_iteration)
            # update direction 
            curr_meta_b_direction = unit_ball_projection(prev_meta_b_direction - b_step_size * full_gradient)
            # update wealth 
            curr_meta_b_wealth = prev_meta_b_wealth - (1 / (self.R * self.L* n_points)) * full_gradient @ prev_meta_b_direction * prev_meta_b_magnitude
            # update betting fraction 
            curr_meta_b_betting_fraction = (1/task_iteration) * ((task_iteration-1) * prev_meta_b_betting_fraction - (1/(self.L * self.R * n_points))* full_gradient @ prev_meta_b_direction) 
            # update magnitude 
            curr_meta_b_magnitude = curr_meta_b_betting_fraction * curr_meta_b_wealth

            self.all_final_weight_vectors.append(np.mean(temp_weight_vectors, axis=0))
            all_test_errors = []
            for idx, curr_test_task in enumerate(data.tr_task_indexes[:task_iteration]):
                all_test_errors.append(loss(data.features_ts[curr_test_task], data.labels_ts[curr_test_task], self.all_final_weight_vectors[idx], loss_name=self.loss_name))
            best_mtl_performances.append(np.nanmean(all_test_errors)) # multi-task learning performance 

        return best_mtl_performances, pd.DataFrame(all_individual_cum_errors).rolling(window=10 ** 10, min_periods=1).mean().values.ravel()






class ParameterFreeConditionalLazyFineTuningVariation:
    def __init__(self, L=1, R=1, K=1, loss_name=None, feature_map_name=None, r=None, W=None):
        ## Assumption Value 
        self.L = L
        self.R = R 
        self.K = K
        ## Feature map parameter 
        self.feature_map_name = feature_map_name
        self.r = r 
        self.W = W
        # wealth 
        self.meta_M_wealth = 1
        self.meta_b_wealth = 1
        self.inner_wealth = 1
        # Loss name 
        self.loss_name = loss_name    
        # Performance Storage
        self.all_individual_cum_errors = []
        self.all_mtl_performances = []
        # Estimated Parameter Storage 
        self.all_final_weight_vectors = []
        self.all_meta_parameters = {"M":[],
                                    "b":[],
                                    "theta":[]}

    def fit(self, data):
        # Initialize meta-parameter M 
        curr_meta_M_fraction = 0 # B 
        curr_meta_M_wealth = self.meta_M_wealth # E = U 
        curr_meta_M_magnitude = curr_meta_M_fraction * curr_meta_M_wealth # P = BU
        mapping = feature_map(data.all_side_info[0], data.labels_tr[0], self.feature_map_name, self.r, self.W)
        curr_meta_M_direction = np.zeros([data.features_tr[0].shape[1], mapping.shape[0]])


        # Initialize meta-parameter b
        curr_meta_b_fraction = 0 # B 
        curr_meta_b_wealth = self.meta_b_wealth # E = U 
        curr_meta_b_magnitude = curr_meta_b_fraction * curr_meta_b_wealth # P = BU
        curr_meta_b_direction = np.zeros(data.features_tr[0].shape[1]) # V 
        
        all_h_meta_M = []
        all_h_meta_b = []
        
        # FOR t = 1, ..., T 
        for task_iteration, task in enumerate(data.tr_task_indexes, 1):
            # Retrive Dataset Z_t = (x_t, y_t)
            x = data.features_tr[task]
            y = data.labels_tr[task]
            # Get side information s_t associated to each task 
            s = data.all_side_info[task] 
            # Get feature Φ(s_t) 
            x_trasf_feature = feature_map(s, y, self.feature_map_name, self.r, self.W)


            # Meta M  
            prev_meta_M_direction = curr_meta_M_direction
            prev_meta_M_fraction = curr_meta_M_fraction
            prev_meta_M_wealth = curr_meta_M_wealth
            prev_meta_M_magnitude = curr_meta_M_magnitude

            # Meta b  
            prev_meta_b_direction = curr_meta_b_direction
            prev_meta_b_fraction = curr_meta_b_fraction
            prev_meta_b_wealth = curr_meta_b_wealth
            prev_meta_b_magnitude = curr_meta_b_magnitude

            ##----------------##
            # Update Meta Para #
            #------------------# 
            # Update transition matrix M 
            meta_M = prev_meta_M_magnitude * prev_meta_M_direction
            self.all_meta_parameters["M"].append(meta_M)

            # Update meta shift b 
            meta_b = prev_meta_b_magnitude * prev_meta_b_direction 
            self.all_meta_parameters["b"].append(meta_b)

            # Update meta parameter theta 
            meta_parameter = meta_M @ x_trasf_feature + meta_b 
            self.all_meta_parameters["theta"].append(meta_parameter)
            
            ## Initialize inner parameter 
            n_points, n_dims = x.shape # n, d
            curr_inner_fraction = 0 # b
            curr_inner_wealth = self.inner_wealth # e
            curr_inner_magnitude = curr_inner_fraction * curr_inner_wealth # p = bu 
            curr_inner_direction = np.zeros(n_dims) # v

            all_h_inner = []

            temp_weight_vectors = [] # to store weight vector obtained within-task
            all_gradient = [] # to store g_{t,i} 
            shuffled_indexes = list(range(n_points))
            # np.random.shuffle(shuffled_indexes) 

            # FOR i = 1, ... , n
            for inner_iteration, curr_point_idx in enumerate(shuffled_indexes, 1):

                # inner  
                prev_inner_direction = curr_inner_direction
                prev_inner_fraction = curr_inner_fraction
                prev_inner_wealth = curr_inner_wealth
                prev_inner_magnitude = curr_inner_magnitude

                # update inner weight vector 
                weight_vector = prev_inner_magnitude * prev_inner_direction + meta_parameter
                # store inner weight vector 
                temp_weight_vectors.append(weight_vector)

                # receive a new data point z_{t,i} = (x_{t,i}, y_{t,i})
                curr_x = x[curr_point_idx, :]
                curr_y = y[curr_point_idx]
                # incur error 
                self.all_individual_cum_errors.append(loss(curr_x, curr_y, weight_vector, loss_name=self.loss_name))

                # compute the gradient 
                subgrad = subgradient(curr_x, curr_y, weight_vector, loss_name=self.loss_name)
                # g_{t,i}
                full_gradient = subgrad * curr_x
                all_gradient.append(full_gradient)

                ##------------------##
                ##   Inner Update   ##
                ##------------------##
                # define inner step size 
                inner_step_size = (1 / (self.L * self.R)) * np.sqrt(2 / inner_iteration)
                # update direction 
                curr_inner_direction = unit_ball_projection(prev_inner_direction - inner_step_size * full_gradient)
                # update wealth 
                curr_inner_wealth = prev_inner_wealth - 1 / (self.R * self.L) * full_gradient @ prev_inner_direction * prev_inner_magnitude
                # update betting fraction via ONS
                g_inner = (1 / (self.R * self.L)) * full_gradient @ prev_inner_direction
                h_inner = g_inner / (1 - g_inner * prev_inner_fraction)
                all_h_inner.append(h_inner)
                A_inner = 1 + np.sum(np.array(all_h_inner) ** 2)
                curr_inner_fraction = np.max([np.min([prev_inner_fraction - (2 / (2 - np.log(3))) * (h_inner / A_inner), 1/2]), -1/2])
                # update magnitude
                curr_inner_magnitude = curr_inner_fraction * curr_inner_wealth

            # compute meta graident G_t = sum g_{t,i} 
            meta_gradient = np.sum(all_gradient, axis=0)

            # receive G_t \Phi(s_t)
            meta_gradient_map = meta_gradient[:,None] @ x_trasf_feature[:,None].T 

            ##------------------##
            ##     M Update     ##
            ##------------------##
            # define meta step size for M 
            M_step_size = (1 / (self.L * self.K * self.R * n_points)) *  np.sqrt(2 / task_iteration)
            # update direction 
            curr_meta_M_direction = unit_ball_projection(prev_meta_M_direction - M_step_size * meta_gradient_map)
            # update wealth 
            curr_meta_M_wealth = prev_meta_M_wealth - (1 / (self.R * self.L * self.K * n_points)) * Frobenius(meta_gradient_map, prev_meta_M_direction) * prev_meta_M_magnitude
            # update betting fraction via Online Newton Step 
            g_meta_M = (1 / (self.L * self.K * self.R * n_points)) * Frobenius(meta_gradient_map, prev_meta_M_direction)
            h_meta_M = g_meta_M / (1 - g_meta_M * prev_meta_M_fraction)
            all_h_meta_M.append(h_meta_M)
            A_meta_M = 1 + np.sum(np.array(all_h_meta_M) ** 2)
            curr_meta_M_fraction = np.max([np.min([prev_meta_M_fraction - (2 / (2 - np.log(3))) * (h_meta_M / A_meta_M), 1 / 2]), - 1 / 2])

            # update magnitude 
            curr_meta_M_magnitude = curr_meta_M_fraction * curr_meta_M_wealth

            ##------------------##
            ##     b Update     ##
            ##------------------##
            # define meta step size for b 
            b_step_size = (1 / (self.L * self.R * n_points)) *  np.sqrt(2 / task_iteration)
            # update direction 
            curr_meta_b_direction = unit_ball_projection(prev_meta_b_direction - b_step_size * full_gradient)
            # update wealth 
            curr_meta_b_wealth = prev_meta_b_wealth - (1 / (self.R * self.L * n_points)) * full_gradient @ prev_meta_b_direction * prev_meta_b_magnitude
            # update betting fraction via ONS 
            g_meta_b = (1 / (self.R * self.L * n_points)) * (meta_gradient @ prev_meta_b_direction)
            h_meta_b = g_meta_b / (1 - g_meta_b * prev_meta_b_fraction)
            all_h_meta_b.append(h_meta_b)
            A_meta_b = 1 + np.sum(np.array(all_h_meta_b) ** 2)
            curr_meta_b_fraction = np.max([np.min([prev_meta_b_fraction - (2 / (2 - np.log(3))) * (h_meta_b / A_meta_b), 1 / 2]), - 1 / 2])
  
            # update magnitude 
            curr_meta_b_magnitude = curr_meta_b_fraction * curr_meta_b_wealth

            self.all_final_weight_vectors.append(np.mean(temp_weight_vectors, axis=0))
            all_test_errors = []
            for idx, curr_test_task in enumerate(data.tr_task_indexes[:task_iteration]):
                all_test_errors.append(loss(data.features_ts[curr_test_task], data.labels_ts[curr_test_task], self.all_final_weight_vectors[idx], loss_name=self.loss_name))
            self.all_mtl_performances.append(np.nanmean(all_test_errors)) # multi-task learning performance 

        return self.all_mtl_performances, pd.DataFrame(self.all_individual_cum_errors).rolling(window=10 ** 10, min_periods=1).mean().values.ravel()
















