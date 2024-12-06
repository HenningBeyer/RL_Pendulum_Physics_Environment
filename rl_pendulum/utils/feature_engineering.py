# An utility file for hording the naming conventions used for the class RL_Cart_Pendulum_Feature_Engineering_Mixin()
# The global naming conventions should be handled separately to the implementation (else just too unreadable)

import numpy as np

class Feature_Engineering_Mapping_Provider():
    """ Note: All mappings provided in this class are strictly tied to the implementation of RL_Cart_Pendulum_Feature_Engineering_Mixin """

    def get_feature_group_to_func_mapping(self):
        """ Returns a mapping of all feature groups to all their corresponding feature group functions (classical feature engineering) """ 
        ## Maps each feature group to a feature group function
        return { "Rod Tip (Cartesian) Positions"        : '_pxr_pyr_features', 
                 "Center of Mass (Cartesian) Positions" : '_pxm_pym_features',
                 "Sine Rod-to-Cart Angles"              : '_sin_thetas',
                 "Cosine Rod-to-Cart Angles"            : '_cos_thetas',
                 "Rod-to-Rod Angles"                    : '_alphas',
                 "Sine Rod-to-Rod Angles"               : '_sin_alphas',
                 "Cosine Rod-to-Rod Angles"             : '_cos_alphas',
                
                 "Average Rod Tip Features (Cartesian/Polar)"               : '_avg_rod_tip_features',
                 "Average Center of Mass Features (Cartesian/Polar)"        : '_avg_com_features',
                 "Rod-Tip-to-Rod-Tip (Polar) Distances"                     : '_rod_to_rod_polar_distances',
                 "Rod-Tip-to-Rod-Tip (Cartesian) Distances"                 : '_rod_to_rod_cartesian_distances',
                 "Rod-Tip-to-Center-of-Mass (Polar) Distances"              : '_rod_to_com_polar_distances', 
                 "Rod-Tip-to-Center-of-Mass (Cartesian) Distances"          : '_rod_to_com_cartesian_distances',
                 "Center-of-Mass-to-Center-of-Mass (Polar) Distances"       : '_com_to_com_polar_distances',
                 "Center-of-Mass-to-Center-of-Mass (Cartesian) Distances"   : '_com_to_com_cartesian_distances',
                 "Energy Features"                                          : '_ernergy_features'
                }
        
    def get_feature_group_to_feature_group_tuple_mapping(self, n):
        """ Returns a mapping to store each tupel for distance feature groups to avoid a step-wise recalculations 
            WARNING: The strings must match the calculation order of the feature group functions!
        """
        n_range = np.arange(1,n+1)
        tup1 = [(n_, n_+offset) for n_ in np.arange(0, n+1) for offset in np.arange(2,n+1) if n_+offset <= n] 
        # --> [(0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (2, 4)] for n=4; polar r-r distance; n(n-1)/2 features
        
        tup2 = [(n_, offset) for n_ in np.arange(0, n+1) for offset in n_range if n_ < offset <= n] 
        # --> [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] for n=3; cartesian r-r distances; 2 * n(n-1) features
        
        # rods tips go from 0 to n, and centers of mass from 1 to n
        ## where (polar) distances of rod tips to their 1 or 2 neighboring center off masses are left out
        tup_start = [(0, n_) for n_ in n_range[1:]]                                                                      # --> [(0, 2), (0, 3), (0, 4)] for n=4
        tup_mid_pos_offset = [(n_, n_+offset) for n_ in n_range[:-1] for offset in n_range[1:] if n_+offset <= n]        # --> [(1, 3), (1, 4), (2, 4)] for n=4
        tup_mid_neg_offset = [(n_, n_+offset) for n_ in n_range[:-1] for offset in np.arange(-n+1, 0) if n_+offset > 0 ] # --> [(2, 1), (3, 1), (3, 2)] for n=4
        tup_mid = tup_mid_pos_offset + tup_mid_neg_offset
        tup_end = [(n, n-offset) for offset in n_range[:-1]]                                                             # --> [(4, 3), (4, 2), (4, 1)] for n=4
        tup3 = tup_start + tup_mid_pos_offset + tup_mid_neg_offset + tup_end
        tup3.sort() # --> [(0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (2, 1), (2, 4), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3)] for n=4; polar r-m distances; (n-1)(n-2)+2(n-1) features        
        
        # rods tips go from 0 to n, and centers of mass from 1 to n
        ## where all (cartesian) distances between rod tips and center of masses are considered
        tup4 = [(n_, n_2) for n_ in np.arange(0,n+1) for n_2 in n_range]
        # --> [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)] for n=3; cartesian r-m distances; 2 * n(n+1)
        
        tup5 = [(n_, n_+offset) for n_ in n_range for offset in n_range if n_+offset <= n] 
        # --> [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)] for n=4; polar m-m distances; n(n-1)/2 features

        tup6 = [(n_, n_+offset) for n_ in n_range for offset in n_range if n_+offset <= n] 
        # --> [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)] for n=4; cartesian m-m distances; 2 * n(n-1)/2 features

        tup7 = [(n_,n_+n_step) for n_step in n_range for n_ in n_range[:-1] if n_+n_step <= n] 
        # --> [(1, 2), (2, 3), (3, 4), (1, 3), (2, 4), (1, 4)] for n=4
                
        return {"Rod-Tip-to-Rod-Tip (Polar) Distances"                     : tup1,
                "Rod-Tip-to-Rod-Tip (Cartesian) Distances"                 : tup2,
                "Rod-Tip-to-Center-of-Mass (Polar) Distances"              : tup3, 
                "Rod-Tip-to-Center-of-Mass (Cartesian) Distances"          : tup4,
                "Center-of-Mass-to-Center-of-Mass (Polar) Distances"       : tup5,
                "Center-of-Mass-to-Center-of-Mass (Cartesian) Distances"   : tup6,
                "Rod-to-Rod Angles"                                        : tup7}
        
    def get_feature_group_to_feature_mapping(self, n):
        n_range = np.arange(1,n+1)
        tupel_mapping = self.get_feature_group_to_feature_group_tuple_mapping(n)
        tup1 = tupel_mapping["Rod-Tip-to-Rod-Tip (Polar) Distances"]
        tup2 = tupel_mapping["Rod-Tip-to-Rod-Tip (Cartesian) Distances"]
        tup3 = tupel_mapping["Rod-Tip-to-Center-of-Mass (Polar) Distances"]
        tup4 = tupel_mapping["Rod-Tip-to-Center-of-Mass (Cartesian) Distances"]
        tup5 = tupel_mapping["Center-of-Mass-to-Center-of-Mass (Polar) Distances"]
        tup6 = tupel_mapping["Center-of-Mass-to-Center-of-Mass (Cartesian) Distances"]
        tup7 = tupel_mapping["Rod-to-Rod Angles"]
        rod_to_rod_polar_distances_strings_ = [f"$$d^{{r-r}}_{{{tup_[0]}, {tup_[1]}}}$$" for tup_ in tup1] 
        rod_to_com_polar_distances_strings_ = [f"$$d^{{r-m}}_{{{tup_[0]}, {tup_[1]}}}$$" for tup_ in tup3] 
        com_to_com_polar_distances_strings_ = [f"$$d^{{m-m}}_{{{tup_[0]}, {tup_[1]}}}$$" for tup_ in tup5] 
        com_to_com_carte_distances_strings_ = [f"$$d^{{m-m}}_{{x_{{{tup_[0]}, {tup_[1]}}}}}$$" for tup_ in tup6] + \
                                              [f"$$d^{{m-m}}_{{y_{{{tup_[0]}, {tup_[1]}}}}}$$" for tup_ in tup6]
        rod_to_rod_carte_distances_strings_ = []
        for tup_ in tup2: # follwoing the order of function _rod_to_rod_cartesian_distances()
            if tup_[0] == 0:
                rod_to_rod_carte_distances_strings_ += [f"$$d^{{r-r}}_{{x_{{{tup_[0]}, {tup_[1]}}}}}$$"]
                # rod_to_rod_carte_distances_strings_ += [f"$$d^{{r-r}}_{{y_{{{tup_[0]}, {tup_[1]}}}}}$$"] # --> this would be one redundant feature: d^rr_y = 0 - pyr = - pyr
            else:
                rod_to_rod_carte_distances_strings_ += [f"$$d^{{r-r}}_{{x_{{{tup_[0]}, {tup_[1]}}}}}$$"] + \
                                                       [f"$$d^{{r-r}}_{{y_{{{tup_[0]}, {tup_[1]}}}}}$$"]
        rod_to_com_carte_distances_strings_ = []
        for tup_ in tup4: # follwoing the order of function _rod_to_com_cartesian_distances()
            if tup_[0] == 0:
                rod_to_com_carte_distances_strings_ += [f"$$d^{{r-m}}_{{x_{{{tup_[0]}, {tup_[1]}}}}}$$"]
                # rod_to_rod_carte_distances_strings_ += [f"$$d^{{r-m}}_{{y_{{{tup_[0]}, {tup_[1]}}}}}$$"] # --> this would be one redundant feature: d^rm_y = 0 - pym = - pym 
            else:
                rod_to_com_carte_distances_strings_ += [f"$$d^{{r-m}}_{{x_{{{tup_[0]}, {tup_[1]}}}}}$$"] + \
                                                       [f"$$d^{{r-m}}_{{y_{{{tup_[0]}, {tup_[1]}}}}}$$"]

        
        energy_strings_ = [fr'$$V_{{{n_}}}$$' for n_ in n_range] + [fr'$$T_{{trans_{{c}}}}$$'] + \
                          [fr'$$T_{{trans_{{{n_}}}}}$$' for n_ in n_range] + \
                          [fr'$$T_{{rot_{n_}}}$$' for n_ in n_range] + \
                          [fr'$$T_{{trans}}$$', fr'$$T_{{rot}}$$', fr'$$V$$', fr'$$T$$', fr'$$L$$']

        # Note: This is a feature-group-to-output-feature mapping!
        return {"Rod Tip (Cartesian) Positions"        : [f"$$p_{{x_{n_}}}^{{r}}$$" for n_ in n_range] + [f"$$p_{{y_{n_}}}^{{r}}$$" for n_ in n_range], 
                "Center of Mass (Cartesian) Positions" : [f"$$p_{{x_{n_}}}^{{m}}$$" for n_ in n_range] + [f"$$p_{{y_{n_}}}^{{m}}$$" for n_ in n_range],
                "Sine Rod-to-Cart Angles"              : [fr"$$\sin(\theta_{{{n_}}})$$" for n_ in n_range],
                "Cosine Rod-to-Cart Angles"            : [fr"$$\cos(\theta_{{{n_}}})$$" for n_ in n_range],
                "Rod-to-Rod Angles"                    : [fr"$$\alpha_{{{tup[0]},{tup[1]}}}$$" for tup in tup7],
                "Sine Rod-to-Rod Angles"               : [fr"$$\sin(\alpha_{{{tup[0]},{tup[1]}}})$$" for tup in tup7],
                "Cosine Rod-to-Rod Angles"             : [fr"$$\cos(\alpha_{{{tup[0]},{tup[1]}}})$$" for tup in tup7],
                
                "Average Rod Tip Features (Cartesian/Polar)"               : [r"$$\bar{p^r_x}$$",        r"$$\bar{p^r_y}$$", 
                                                                              r"$$\bar{\theta_{r-c}}$$", r"$$\bar{p^d_{r-c, x}}$$", 
                                                                              r"$$\bar{d_{r-c}}$$"],
                "Average Center of Mass Features (Cartesian/Polar)"        : [r"$$\bar{p^m_x}$$"       , r"$$\bar{p^m_y}$$", 
                                                                              r"$$\bar{p^{m*}_x}$$"    , r"$$\bar{p^{m*}_y}$$",
                                                                              r"$$\bar{\theta_{m-c}}$$", r"$$\bar{\theta_{m*-c}}$$", 
                                                                              r"$$\bar{p^d_{m-c, x}}$$", r"$$\bar{p^d_{m*-c, x}}$$", 
                                                                              r"$$\bar{d_{m-c}}$$",      r"$$\bar{d_{m*-c}}$$"],
                "Rod-Tip-to-Rod-Tip (Polar) Distances"                     : rod_to_rod_polar_distances_strings_,
                "Rod-Tip-to-Rod-Tip (Cartesian) Distances"                 : rod_to_rod_carte_distances_strings_,
                "Rod-Tip-to-Center-of-Mass (Polar) Distances"              : com_to_com_polar_distances_strings_, 
                "Rod-Tip-to-Center-of-Mass (Cartesian) Distances"          : rod_to_com_carte_distances_strings_,
                "Center-of-Mass-to-Center-of-Mass (Polar) Distances"       : com_to_com_polar_distances_strings_,
                "Center-of-Mass-to-Center-of-Mass (Cartesian) Distances"   : com_to_com_carte_distances_strings_,
                "Energy Features"                                          : energy_strings_}  
        
    def get_feature_to_feature_group_mapping(self, n):
        return {v__ : k_ for k_, v_ in self.get_feature_group_to_feature_mapping(n).items() for v__ in v_}
    
    def get_func_to_output_feat_mapping(self, n):
        """ Returns a function-to-output-feature_names mapping. Gives information which functions return which group of single features. 
            The feature string names are needed for code clarity (indexing by string, not by number).
        """
        fg2func  = self.get_feature_group_to_func_mapping()
        fg2feats = self.get_feature_group_to_feature_mapping(n)
        return {fg2func[fgroup_] : fg2feats[fgroup_] for fgroup_ in fg2func.keys()}
    
    # Custom lookback windows for time series feature engineering:
    ## (they will have to be specifically chosen for each problem)
    ## Note that a lookback window of 2 seconds means 2 s / 1 ms = 2000 steps; 2 s / (1*10^(-6) s) = 2000000 steps! inside the replay buffer    
    
    ## max lookback time: to be 2 s
    ## min lookback time: to be 0.01 s
    def get_non_windowed_offsets(self):
        return {'long'   : np.array([0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]),
                'medium' : np.array([0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25]),
                'short'  : np.array([0.01, 0.025, 0.05, 0.075, 0.1])}
    
    def get_overlapping_windowed_offsets(self):
        return {'long'   : np.array([0.05, 0.125, 0.25, 0.5, 1.0, 1.5, 2.0]),
                'medium' : np.array([0.05, 0.125, 0.25, 0.5]),
                'short'  : np.array([0.05, 0.125])}
        
    def get_non_overlapping_windowed_offsets(self):
        # Note: the first step should be chosen like the overlapping_windowed_offsets so that features can be reused
        return {'long'   : np.array([0.05, 0.125, 0.25, 0.5, 1.0, 1.5, 2.0]), 
                'medium' : np.array([0.05, 0.125, 0.25, 0.5]),
                'short'  : np.array([0.05, 0.125])}
        
    def get_min_max_lookback_time(self):
        """ Returns the times for the smallest and biggest look-back window. This is currently only used for sanity checking
            The values change by specifying different look-back mappings above.
        """
        return 0.01, 2.0 # in s
    
class Feature_Engineering_Param_Provider():
    """ class which holds all sectable options of feature groups and single features for CFE, DFE ans TSFE 

        This class is called by the UI to fill the MultiSelect boxes.
        This class is called by the Param manager classes to validate the selections.
    """
    def __init__(self):
        pass # config holds no parameters yet, just a method provider
       
    def get_all_feature_groups(self):
        """ This is for copy-pasting """
        return  ["Sine Rod-to-Cart Angles", "Cosine Rod-to-Cart Angles", "Rod-to-Rod Angles", "Sine Rod-to-Rod Angles", "Cosine Rod-to-Rod Angles",
                 "Average Rod Tip Features (Cartesian/Polar)", "Average Center of Mass Features (Cartesian/Polar)",
                 "Rod-Tip-to-Rod-Tip (Polar) Distances", 'Center-of-Mass-to-Center-of-Mass (Polar) Distances', "Rod-Tip-to-Center-of-Mass (Polar) Distances",
                 "Rod-Tip-to-Rod-Tip (Cartesian) Distances", "Center-of-Mass-to-Center-of-Mass (Cartesian) Distances", "Rod-Tip-to-Center-of-Mass (Cartesian) Distances",
                 "Rod Tip (Cartesian) Positions", "Center of Mass (Cartesian) Positions", 
                 "Energy Features"]

    def get_all_tsfe_features(self, n):
        """ Useful for copy and pasting """
        n_range = np.arange(1,n+1)

        x_c_cols     = ['$$x_c$$', r'$$\dot{x_c}$$', r'$$\ddot{x_c}$$']
        theta_cols   = [fr'$$\theta_{{{n_}}}$$' for n_ in n_range]
        dtheta_cols  = [fr'$$\dot{{\theta_{n_}}}$$' for n_ in n_range]
        ddtheta_cols = [fr'$$\ddot{{\theta_{n_}}}$$' for n_ in n_range]
        pxr_cols     = [fr"$$p_{{x_{n_}}}^{{r}}$$" for n_ in n_range] 
        pyr_cols     = [fr"$$p_{{y_{n_}}}^{{r}}$$" for n_ in n_range]

        sin_theta_cols = [fr'$$\sin(\theta_{{{n_}}})$$' for n_ in n_range]
        cos_theta_cols = [fr'$$\cos(\theta_{{{n_}}})$$' for n_ in n_range]
        alpha_tuples   = [(n_,n_+n_step) for n_step in n_range for n_ in n_range[:-1] if n_+n_step <= n] # --> [(1, 2), (2, 3), (3, 4), (1, 3), (2, 4), (1, 4)] for n=4
        alpha_cols     = [fr'$$\alpha_{{{tup_[0]},{tup_[1]}}}$$' for tup_ in alpha_tuples] 
        sin_alpha_cols = [fr'$$\sin(\alpha_{{{tup_[0]},{tup_[1]}}})$$' for tup_ in alpha_tuples] 
        cos_alpha_cols = [fr'$$\cos(\alpha_{{{tup_[0]},{tup_[1]}}})$$' for tup_ in alpha_tuples] 

        return ['$$r$$'] + x_c_cols + theta_cols + dtheta_cols + ddtheta_cols + pxr_cols + pyr_cols +\
                    sin_theta_cols + cos_theta_cols + alpha_cols + sin_alpha_cols + cos_alpha_cols
 
    def get_base_features(self, n):
        n_range = np.arange(1,n+1)
        x_c_cols     = ['$$x_c$$', r'$$\dot{x_c}$$', r'$$\ddot{x_c}$$']
        theta_cols   = [fr'$$\theta_{{{n_}}}$$' for n_ in n_range]
        dtheta_cols  = [fr'$$\dot{{\theta_{n_}}}$$' for n_ in n_range]
        ddtheta_cols = [fr'$$\ddot{{\theta_{n_}}}$$' for n_ in n_range]
        return  ['$$r$$'] + x_c_cols + theta_cols + dtheta_cols + ddtheta_cols  
        
              
    def notes(self):      
        """ This is just for copy pasting: """       
        fwin_func_groups     = ['Mean', 'Min', 'Max', 'Std']
        now_fwin_func_groups = ['Now-Mean', 'Now-Min', 'Now-Max']
        firstfwin_fwin_func_groups = ['Mean-Mean', 'Mean-Min', 'Mean-Max',
                                        'Min-Mean',  'Min-Min',  'Min-Max',
                                        'Max-Mean',  'Max-Min',  'Max-Max',
                                        'Std-Std']