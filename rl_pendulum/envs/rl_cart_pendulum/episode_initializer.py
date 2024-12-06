import numpy as np

class Episode_Initializer:
    """ This class is used for deterministic episode initializing. """
    def __init__(self, **params):
        self.sample_mapping = {
            'T'                     : params['T'],
            'initial_cart_pos'      : params['initial_cart_pos'],
            'initial_cart_velocity' : params['initial_cart_velocity'],
            'initial_angles'        : params['initial_angles'],
            'initial_velocities'    : params['initial_velocities'],
        }
    
    def sample(self):
        return self.sample_mapping
    
class Random_Episode_Initializer:   
    
    def __init__(self, **params):
        """ - Using a random episode initializer is always recommended, as it provides much higher quality training data
            - A normal distribution is used for sampling with a bell shape centered at the interval mean. 
            - All sampled values will lie inside the interval without outliers by using np.clip(). 
            - To chose a non-random value, simply specify single-value intervals like [10, 10] or [1.25, 1.25].
        """
        self.n = params['n']
        self.sample_mapping = None                                
                                        
        self.param_mapping = {
            'T'                     : [params['T_interval'],                       1, 'int'],
            'initial_cart_pos'      : [params['initial_cart_pos_interval'],        1, 'int'],
            'initial_cart_velocity' : [params['initial_cart_velocity_interval'],   1, 'int'],
            'initial_angles'        : [params['initial_angles_interval'],     self.n, 'arr'],
            'initial_velocities'    : [params['initial_velocities_interval'], self.n, 'arr'],
        }
        
    def _single_sample(self, interval, num_samples=1, dtype='arr'):
        mean    = (interval[0] + interval[1])/2  
        std_dev = (interval[1] - interval[0])/4.5  # This can be used to adjust the bell-shape width
                                                   # (upper_bound - lower_bound)/4.5, was found to the yield best bell shape
        sample = np.random.normal(loc=mean, scale=std_dev, size=num_samples)
        sample = np.clip(sample, interval[0], interval[1]) # clip so that all samples lie within [lower_bound, upper_bound]
        if dtype == 'int':
            return sample[0]
        elif dtype == 'arr':
            pass # nothing to be done
        return sample
    
    def sample(self):
        self.sample_mapping = {param_str_ : self._single_sample(interval=params_[0], num_samples=params_[1], dtype=params_[2]) 
                                   for param_str_, params_ in self.param_mapping.items()}
        return self.sample_mapping # --> {'T' : 11.3, 'initial_cart_pos' : -0.16, ..., 'initial_angles' : [0.324, -3.112, -2.11], ...}
    