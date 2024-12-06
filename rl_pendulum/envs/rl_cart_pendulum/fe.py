import numpy as np
from rl_pendulum.utils.feature_engineering import Feature_Engineering_Mapping_Provider

class RL_Cart_Pendulum_Feature_Engineering():
    """ A full feature engineering class that features:
            CFE: classical feature engineering, +,-,*,/,... operations between features at time-stamp t
            DFE: derivative feature engineering w.r.t. t using the central difference method; minorly delayed by 2 time steps (needs past 3 time stamps)
            TSFE: time series feature engineering separatly for every feature in a window and aggregation functions like mean(), min(), max(), std()
                - needs 2s/dt time stamps ~2000 for dt=0.001.
                - Substeps:
                    - TS differences: simply calculates differences a_now - a_later for any selected feature
                    - overlapping windows: applies aggregation functions over windows starting all at a_now and stretch back in different amounts
                    - non-overlapping windows: applies aggregation functions over windows that lie consecutively behind each other without gaps
                    
        The feature engineering was tied to a replay buffer with pre-allocated memory (neccessary for good performance when updating)
        --> the whole buffer is passed by reference, even the values of pre-allocated memory that will be set on later steps
        --> so indexing always starts from buffer_idx

    """
    
    def reset_feature_engineering(self):
        """ This is a builder method called for every env.reset() """
        self._init_tsfe_lookback_windows()
        self._initialize_classical_mappings()
        self._initialize_mappings()
        self._build_fe_pipeline()
    
    def _initialize_classical_mappings(self):
        mapping_provider = Feature_Engineering_Mapping_Provider()
        self.fgroup2func = mapping_provider.get_feature_group_to_func_mapping()
        self.fgroup2tup  = mapping_provider.get_feature_group_to_feature_group_tuple_mapping(self.n)
        self.fgroup2feat = mapping_provider.get_feature_group_to_feature_mapping(self.n)
        self.feat2fgroup = mapping_provider.get_feature_to_feature_group_mapping(self.n)
        self.func2feat   = mapping_provider.get_func_to_output_feat_mapping(self.n)
        
    def _init_tsfe_lookback_windows(self):
        """ Called with RL_Cart_Pendulum_Environment_Core.reset(). """
        mapping_provider = Feature_Engineering_Mapping_Provider()
        # gets the step amount array for each lookback window
        if self.do_difference_tsfe:
            self.non_windowed_offsets = np.round(mapping_provider.get_non_windowed_offsets()[self.tsfe_lookback_mode]/self.dt).astype(int)          # [0.01, ..., 2.0] --> [10, ..., 2000]
        else:
            self.non_windowed_offsets = np.array([])
            
        if self.do_overlapping_win_tsfe:
            self.overlapping_windowed_offsets  = np.round(mapping_provider.get_overlapping_windowed_offsets()[self.tsfe_lookback_mode]/self.dt).astype(int)      # [0.01, ..., 2.0] --> [10, ..., 2000]
        else:
            self.overlapping_windowed_offsets  = np.array([])
        
        if self.do_non_overlapping_win_tsfe:
            self.non_overlapping_windowed_offsets  = np.round(mapping_provider.get_non_overlapping_windowed_offsets()[self.tsfe_lookback_mode]/self.dt).astype(int)  # [0.01, ..., 2.0] --> [10, ..., 2000]
        else:
            self.non_overlapping_windowed_offsets  = np.array([])
            
        min_look_back_time, max_look_back_time = mapping_provider.get_min_max_lookback_time()
        assert (min_look_back_time >= self.dt), (f'self.dt = {self.dt} is too big for min_look_back_time = {min_look_back_time}. Decrease dt or increase min_look_back_time')

        
    def _initialize_mappings(self):
        """ Function that handles the complete process of mapping initializion for feature engineering
            There have to be that many mappings for flexibly adjustable pipelines, that stay efficient during inference. 
        """  

        base_feats            = self.base_col_names # set in RL_Cart_Pendulum_Environment_Core for convenience
        _classic_funcs        = [self.fgroup2func[key_]  for key_ in self.feature_groups]
        _classic_feats_nested = [self.func2feat[key_]  for key_ in _classic_funcs]
        if _classic_feats_nested == []: # if no feature group was chosen to do CFE
            classic_feats = _classic_feats_nested
        else:
            classic_feats = np.concatenate([self.func2feat[key_]  for key_ in _classic_funcs])
        
        _dfe_classic_funcs        = [self.fgroup2func[key_]  for key_ in self.dfe_feature_groups] # differentiated features are always a subset of the base features
        self.differentiated_feats = np.concatenate([self.func2feat[key_]  for key_ in _dfe_classic_funcs] + [self.dfe_included_base_features]) 
        if len(self.differentiated_feats) != 0: # if != []
            derivative_feats = [[fr"$$\frac{{\partial^{ord_}}}{{\partial\mathit{{t}}^{ord_}}}" + feat_.replace('$$', '', 1) for feat_ in self.differentiated_feats] 
                                    for ord_ in np.arange(1,self.max_derivative_order+1)]
        else:
            derivative_feats = [[]] # [[]] allows np.concatenate to be applied like on the other case

        """ The mappings of time series feature engineering are a bit of a pain, but they separate naming conventions from the calculations, 
            which is more readable and faster per step
        """
        self.aggfstr2func = {'Mean' : np.mean, 'Min' : np.min, 'Max' : np.max, 'Std' : np.std}
        # self.tsfe_aggfuncs = {'4.2' : [aggfstr2func[fstr_] for fstr_ in self.overlapping_win_agg_funcs],
        #                       '4.3' : [aggfstr2func[fstr_] for fstr_ in self.non_overlapping_win_agg_funcs]}
        self._start_end_offsets_4_3 = list(zip(np.concatenate([[0], self.non_overlapping_windowed_offsets[:-1]]), self.non_overlapping_windowed_offsets)) # used as an interator; use list(zip()) to make it infinite
        
        # These mappings are a bit of a pain, but they separate naming conventions from the calculations, which is more readable, debuggable, and it is faster per step
        if self.do_difference_tsfe:
            tsfe_4_1_diff_mapping       = {(offset_, '4.1 diff') : [fr"$$\text{{Diff}}({feat_.replace('$$', '')},{offset_})$$" 
                                                        for feat_ in self.tsfe_features] 
                                                for offset_ in self.non_windowed_offsets}
        else:
            tsfe_4_1_diff_mapping = {}
        
        if self.do_overlapping_win_tsfe:
            tsfe_4_2_fwin_mapping       = {(f_str_, offset_, '4.2 fwin') : [fr"$$\text{{{f_str_}}}({feat_.replace('$$', '')},{offset_})$$" 
                                                                    for feat_ in self.tsfe_features] 
                                                for offset_ in self.overlapping_windowed_offsets
                                                    for f_str_ in self.overlapping_fwin_aggfuncs}
            
            _tsfe_4_2_now_fwin_aggfuncs = [str_.replace('Now-', '') for str_ in self.overlapping_now_fwin_aggfuncs]
            tsfe_4_2_now_fwin_mapping   = {(f_str_, offset_, '4.2 now-fwin') : [fr"$${feat_.replace('$$', '')} - \text{{{f_str_}}}({feat_.replace('$$', '')},{offset_})$$" 
                                                                    for feat_ in self.tsfe_features] 
                                                for offset_ in self.overlapping_windowed_offsets
                                                    for f_str_ in _tsfe_4_2_now_fwin_aggfuncs}
            
            _tsfe_4_2_ffwin_fwin_aggfuncs_left  = [str_[:str_.find('-')  ] for str_ in self.overlapping_ffwin_fwin_aggfuncs] # 'Mean-Max' --> 'Mean'
            _tsfe_4_2_ffwin_fwin_aggfuncs_right = [str_[str_.find('-')+1:] for str_ in self.overlapping_ffwin_fwin_aggfuncs] # 'Mean-Max' --> 'Max'
            tsfe_4_2_ffwin_fwin_mapping = {(f_str_left_, f_str_right_, offset_, '4.2 ffwin-fwin') : 
                                                            [fr"$$\text{{{f_str_left_}}}({feat_.replace('$$','')},{self.non_windowed_offsets[0]}) - " + \
                                                                fr"\text{{{f_str_right_}}}({feat_.replace('$$','')},{offset_}$$"  
                                                                    for feat_ in self.tsfe_features] 
                                                for offset_ in self.overlapping_windowed_offsets
                                                    for f_str_left_, f_str_right_ in zip(_tsfe_4_2_ffwin_fwin_aggfuncs_left, _tsfe_4_2_ffwin_fwin_aggfuncs_right)
                                                        if offset_ != self.overlapping_windowed_offsets[0]}
        else:
            tsfe_4_2_fwin_mapping       = {}
            tsfe_4_2_now_fwin_mapping   = {}
            tsfe_4_2_ffwin_fwin_mapping = {}
        
        if self.do_non_overlapping_win_tsfe:
            tsfe_4_3_fwin_mapping       = {(f_str_, start_, end_, '4.3 fwin') : [fr"$$\text{{{f_str_}}}({feat_.replace('$$', '')},{start_},{end_})$$"
                                                                    for feat_ in self.tsfe_features] 
                                                for start_, end_ in self._start_end_offsets_4_3
                                                    for f_str_ in self.non_overlapping_fwin_aggfuncs}
            
            _tsfe_4_3_now_fwin_aggfuncs = [str_.replace('Now-', '') for str_ in self.non_overlapping_now_fwin_aggfuncs]
            tsfe_4_3_now_fwin_mapping   = {(f_str_, start_, end_, '4.3 now-fwin') : [fr"$${feat_.replace('$$', '')} - \text{{{f_str_}}}({feat_.replace('$$', '')},{start_},{end_})$$" 
                                                                    for feat_ in self.tsfe_features] 
                                                for start_, end_ in self._start_end_offsets_4_3
                                                    for f_str_ in _tsfe_4_3_now_fwin_aggfuncs}
            
            _tsfe_4_3_ffwin_fwin_aggfuncs_left  = [str_[:str_.find('-')  ] for str_ in self.non_overlapping_ffwin_fwin_aggfuncs] # 'Mean-Max' --> 'Mean'
            _tsfe_4_3_ffwin_fwin_aggfuncs_right = [str_[str_.find('-')+1:] for str_ in self.non_overlapping_ffwin_fwin_aggfuncs] # 'Mean-Max' --> 'Max'
            tsfe_4_3_ffwin_fwin_mapping = {(f_str_left_, f_str_right_, start_, end_, '4.3 ffwin-fwin') : 
                                                                    [ fr"$$\text{{{f_str_left_}}}({feat_.replace('$$','')},{0},{self.non_windowed_offsets[0]}) - " + \
                                                                    fr"\text{{{f_str_right_}}}({feat_.replace('$$','')},{start_},{end_}$$"  
                                                                    for feat_ in self.tsfe_features] 
                                                for start_, end_ in self._start_end_offsets_4_3
                                                    for f_str_left_, f_str_right_ in zip(_tsfe_4_3_ffwin_fwin_aggfuncs_left, _tsfe_4_3_ffwin_fwin_aggfuncs_right)
                                                        if start_ != 0}
        else:
            tsfe_4_3_fwin_mapping       = {}
            tsfe_4_3_now_fwin_mapping   = {}
            tsfe_4_3_ffwin_fwin_mapping = {}
        
        tsfe_feats   = np.concatenate([np.ravel(list(tsfe_4_1_diff_mapping.values())), # list() in case of dict_value object
                                       np.ravel(list(tsfe_4_2_fwin_mapping.values())), # np.ravel converts n-dim arrays to 1-dim arrays
                                       np.ravel(list(tsfe_4_2_now_fwin_mapping.values())), 
                                       np.ravel(list(tsfe_4_2_ffwin_fwin_mapping.values())),
                                       np.ravel(list(tsfe_4_3_fwin_mapping.values())), 
                                       np.ravel(list(tsfe_4_3_now_fwin_mapping.values())), 
                                       np.ravel(list(tsfe_4_3_ffwin_fwin_mapping.values()))])

        all_feats        = np.concatenate([np.ravel(list(base_feats)), # base_feats need to be listed first!
                                           np.ravel(list(classic_feats)), 
                                           np.ravel(list(derivative_feats)), 
                                           np.ravel(list(tsfe_feats))]) 
        self.s2i         = {feat_ : col_idx_ for col_idx_, feat_ in enumerate(all_feats)}
        self.derivative_input_indizes  = np.array([self.s2i[feat_] for feat_ in self.differentiated_feats])
        self.derivative_output_indizes = {ord_ : np.array([self.s2i[feat_] for feat_ in derivative_feats[ord_-1]]) for ord_ in np.arange(1,self.max_derivative_order+1)}
        
        self.tsfe_input_idxlist = [self.s2i[feat_] for idx_, feat_ in enumerate(self.tsfe_features)]
        d1 = {k : [self.s2i[feat_] for feat_ in v] for k, v in tsfe_4_1_diff_mapping.items()}
        d2 = {k : [self.s2i[feat_] for feat_ in v] for k, v in tsfe_4_2_fwin_mapping.items()}
        d3 = {k : [self.s2i[feat_] for feat_ in v] for k, v in tsfe_4_2_now_fwin_mapping.items()}
        d4 = {k : [self.s2i[feat_] for feat_ in v] for k, v in tsfe_4_2_ffwin_fwin_mapping.items()}
        d5 = {k : [self.s2i[feat_] for feat_ in v] for k, v in tsfe_4_3_fwin_mapping.items()}
        d6 = {k : [self.s2i[feat_] for feat_ in v] for k, v in tsfe_4_3_now_fwin_mapping.items()}
        d7 = {k : [self.s2i[feat_] for feat_ in v] for k, v in tsfe_4_3_ffwin_fwin_mapping.items()}
        self.tsfe_output_key2idxlist = {**d1, **d2, **d3, **d4, **d5, **d6, **d7} 
        # --> call self.tsfe_output_key2idxlist['Mean', 2000, '4.2 fwin'] or self.tsfe_output_key2idxlist['Mean', 2000, 2500, '4.3 fwin'] for an index list
            # other example: self.tsfe_output_key2idxlist['Mean', 'Max', 2000, 2500, '4.3 ffwin-fwin']
            
    def _build_fe_pipeline(self):
        """ Builds a sequential pipeline from the initlized mappings. 
            - This pipeline is sequential, as threading and multiprocessing will not yield great benefits
            - vectorization is used for derivative features and for time series feature engineering as far as possible
            - all functions have to take arr as input
            - all functions only support step-wise feature engineering
            A pipeline gives overall more flexible code, and clarity through abstraction
        """
        self.pipe = []
        if self.do_feature_engineering:
            if self.do_classic_engineering:
                self.pipe += list(np.ravel([getattr(self, self.fgroup2func[key_])  for key_ in self.feature_groups])) # ['Sine Rod-to-Cart Angles'] --> ['sin_thetas'] --> [<func_obj> of RL_Cart_Pendulum_Feature_Engineering]
            if self.do_derivative_engineering:
                self.pipe += [self._calc_state_function_derivatives]
            if self.do_ts_feature_engineering:
                if self.do_difference_tsfe:
                    self.pipe += [self._time_series_differences]
                if self.do_overlapping_win_tsfe:
                    self.pipe += [self._overlapping_window_tsfe]
                if self.do_non_overlapping_win_tsfe:
                    self.pipe += [self._non_overlapping_window_tsfe]            
            self.pipe += [self._post_process_data]
            
    def run_fe_pipeline(self, arr : np.array, buffer_idx : int):
        """ Function which simply runs the pipeline sequentially"""    
        for func_ in self.pipe:
            func_(arr, buffer_idx)
            
    def _calc_state_function_derivatives(self, arr : np.array, buffer_idx : int):
        """ - Data of at max 5 time steps is needed!
            - Differentiating noisy features yields spiky features and amplifies noise
            - Having analog signals should be differentiated at max only once; these features are usable for agent control, as these clear spikes can alter agent bahviour faster
            - Differentiating against other features (they would also be unevenly spaced) results in many singularities 
                --> The reason are maxima and noise (the denominator can get zero); almost any feature will have noise or extrema, and hence any feature will have such singularities quite often.
                --> So using any differentiation features except t, is NOT recommended
                
            The central difference approximation Formulas are used to derive numerical n-th order derivatives:
            Sources: [https://www.youtube.com/watch?v=Tfo12ylAMso&t=484s, https://www.youtube.com/watch?v=9fGaTU1-f-0]
            Formulas: https://math.stackexchange.com/questions/3702607/central-difference-approximations
            - The central difference method would needs future time stamps
            --> so the data gets delayed/lagged by some amount of steps to still calculate the derivatives relatively accurate
            --> 2 nan values for the first 2 steps
        """
        if self.max_derivative_order >= 1 and buffer_idx >= 2: # This might be the maximum order for noisy features!
            # calculation lagged by one step
            arr[buffer_idx, self.derivative_output_indizes[1]] = (arr[buffer_idx,   self.derivative_input_indizes] - \
                                                                  arr[buffer_idx-2, self.derivative_input_indizes])/(2*self.dt)
        else:
            arr[buffer_idx, self.derivative_output_indizes[1]] = 0 # this is the np.nan replacement; just visible for 2 steps
        
        if self.max_derivative_order >= 2 and buffer_idx >= 2: # very noisy on even tiniest noise!
            # calculation lagged by one step
            arr[buffer_idx, self.derivative_output_indizes[2]] = (arr[buffer_idx,   self.derivative_input_indizes] - \
                                                                2*arr[buffer_idx-1, self.derivative_input_indizes] + \
                                                                  arr[buffer_idx-2, self.derivative_input_indizes])/(self.dt**2)
        else:
            arr[buffer_idx, self.derivative_output_indizes[1]] = 0 # this is the np.nan replacement; just visible for 2 steps
        
        # 3rd and 4rd derivatives never recomended, even without noise 
    def _time_series_differences(self, arr : np.array, buffer_idx : int):
        """ Step 4.1: Non-Windowed Momentum Features
            features: [now - later]
            The implementation is partially parallized.
        """     
        for offset_ in self.non_windowed_offsets:
            offset__ = min(offset_, self.step_num) # implementing a dynamic window (less np.nan values)
            arr[buffer_idx, self.tsfe_output_key2idxlist[(offset_, '4.1 diff')]] = \
                arr[buffer_idx, self.tsfe_input_idxlist] - arr[buffer_idx - offset__, self.tsfe_input_idxlist]

    def _overlapping_window_tsfe(self, arr : np.array, buffer_idx : int):
        """ Step 4.2: Moving Overlapping Window Features
        
            functions: [mean(), min(), max(), std()] 
            features: [f_win, now - f_win, first_f_win - f_win] 
            now - f_win excludes:         [std()]
            first_f_win - f_win includes: [mean()-mean, min()-min(), max()-max(), std()-std(), mean()-min(), mean()-max(), min()-max()] 
            
            The aggregation windows all stretch n steps back from the most recent time step, and thus are mostly overlapping
            The implementation is partially parallized.
        """
        # [f_win]
        for agg_func_str_ in self.overlapping_fwin_aggfuncs: 
            agg_func_ = self.aggfstr2func[agg_func_str_]
            for offset_ in self.overlapping_windowed_offsets:
                offset__ = min(offset_, self.step_num) # implementing a dynamic window (less np.nan values)
                arr[buffer_idx, self.tsfe_output_key2idxlist[(agg_func_str_, offset_, '4.2 fwin')]] = \
                    agg_func_(arr[buffer_idx - offset__:buffer_idx+1, self.tsfe_input_idxlist]) 
                
                
        # [now - f_win]
        for agg_func_str_ in self.overlapping_now_fwin_aggfuncs: 
            agg_func_str_ = agg_func_str_.replace('Now-', '')
            agg_func_ = self.aggfstr2func[agg_func_str_]
            for offset_ in self.overlapping_windowed_offsets:      
                offset__ = min(offset_, self.step_num) # implementing a dynamic window (less np.nan values)
                arr[buffer_idx, self.tsfe_output_key2idxlist[(agg_func_str_, offset_, '4.2 now-fwin')]] = \
                    arr[buffer_idx, self.tsfe_input_idxlist] - agg_func_(arr[buffer_idx - offset__:buffer_idx+1, self.tsfe_input_idxlist]) 
             
        # [first_f_win - f_win]
        offset_smallest = self.overlapping_windowed_offsets[0] 
        offset_smallest__ = min(offset_smallest, self.step_num) # implementing a dynamic window (less np.nan values)
        for agg_func_str_ in self.overlapping_ffwin_fwin_aggfuncs: 
            str_l_, str_r_           = agg_func_str_[:agg_func_str_.find('-')  ], agg_func_str_[agg_func_str_.find('-')+1: ]
            agg_func_l_, agg_func_r_ = self.aggfstr2func[str_l_], self.aggfstr2func[str_r_], 
            for offset_ in self.overlapping_windowed_offsets[1:]:  
                if offset_ == offset_smallest:
                    continue
                offset__ = min(offset_, self.step_num) # implementing a dynamic window (less np.nan values)
                arr[buffer_idx, self.tsfe_output_key2idxlist[(str_l_, str_r_, offset_, '4.2 ffwin-fwin')]] = \
                    agg_func_l_(arr[buffer_idx - offset_smallest__:buffer_idx+1, self.tsfe_input_idxlist]) - \
                    agg_func_r_(arr[buffer_idx - offset__         :buffer_idx+1, self.tsfe_input_idxlist]) 
        
    def _non_overlapping_window_tsfe(self, arr : np.array, buffer_idx : int):
        """ Step 4.3: Moving Non-Overlapping Window Features
        
            functions: [mean(), min(), max(), std()] 
            features: [f_win, now - f_win, first_f_win - f_win] 
            now - f_win excludes:         [std()]
            first_f_win - f_win includes: [mean()-mean, min()-min(), max()-max(), std()-std(), mean()-min(), mean()-max(), min()-max()] 
            
            The aggregation windows lie consecutively behind each other without gaps or overlapping. Each window starts behind the previous window.
            The implementation is partially parallized.
        """
        # [f_win]        
        for agg_func_str_ in self.non_overlapping_fwin_aggfuncs: 
            agg_func_ = self.aggfstr2func[agg_func_str_]
            for start_, end_ in self._start_end_offsets_4_3:
                start__ = max(buffer_idx-start_, buffer_idx-self.step_num)
                end__   = max(buffer_idx-end_, buffer_idx-self.step_num)
                arr[buffer_idx, self.tsfe_output_key2idxlist[(agg_func_str_, start_, end_, '4.3 fwin')]] = \
                    agg_func_(arr[end__:start__+1, self.tsfe_input_idxlist]) 
         
        # [now - f_win]
        for agg_func_str_ in self.non_overlapping_now_fwin_aggfuncs:
            agg_func_str_ = agg_func_str_.replace('Now-', '')
            agg_func_ = self.aggfstr2func[agg_func_str_]
            for start_, end_ in self._start_end_offsets_4_3:     
                start__ = max(buffer_idx-start_, buffer_idx-self.step_num)
                end__   = max(buffer_idx-end_, buffer_idx-self.step_num)
                arr[buffer_idx, self.tsfe_output_key2idxlist[(agg_func_str_, start_, end_, '4.3 now-fwin')]] = \
                    arr[buffer_idx, self.tsfe_input_idxlist] - \
                    agg_func_(arr[end__:start__+1, self.tsfe_input_idxlist]) 
            
        # [first_f_win - f_win]
        end_0 = self.non_overlapping_windowed_offsets[0]        
        for agg_func_str_ in self.non_overlapping_ffwin_fwin_aggfuncs: 
            str_l_, str_r_           = agg_func_str_[:agg_func_str_.find('-')  ], agg_func_str_[agg_func_str_.find('-')+1: ]
            agg_func_l_, agg_func_r_ = self.aggfstr2func[str_l_], self.aggfstr2func[str_r_]
            for start_, end_ in self._start_end_offsets_4_3: 
                if start_ == 0:
                    continue
                start__ = max(buffer_idx-start_, buffer_idx-self.step_num)
                end__   = max(buffer_idx-end_,   buffer_idx-self.step_num)
                end0__  = max(buffer_idx-end_0,  buffer_idx-self.step_num)
                arr[buffer_idx, self.tsfe_output_key2idxlist[(str_l_, str_r_, start_, end_, '4.3 ffwin-fwin')]] = \
                    agg_func_l_(arr[end0__:buffer_idx+1, self.tsfe_input_idxlist]) - \
                    agg_func_r_(arr[end__ :start__+1,    self.tsfe_input_idxlist]) 

    def _post_process_data(self, arr : np.array, buffer_idx : int):
        pass #
        # replace infinity by np.nan
        #df.replace([np.inf, -np.inf], np.nan, inplace=True) 
        # fill any nan with 0
        #df.fillna(value=0, inplace=True) # there should not be nan values. If singularities happen, errors should be thrown later

        
    
    ### Support features
    def __pxr_pyr_features(self, arr : np.array, buffer_idx : int):
        # Function will only work for step-wise calculations, ie arr[buffer_idx, ...]
        l_list       = np.array([self.constants['str_to_val'][f"l_{n_}"] for n_ in self.n_range])
        pxr_summands = np.array([-l_list[n_-1]*np.sin(arr[buffer_idx, self.s2i[fr"$$\theta_{{{n_}}}$$"]]) for n_ in self.n_range])
        pyr_summands = np.array([ l_list[n_-1]*np.cos(arr[buffer_idx, self.s2i[fr"$$\theta_{{{n_}}}$$"]]) for n_ in self.n_range])
        out_pxr = np.array([[np.sum(pxr_summands[:n_], axis=0) for n_ in self.n_range]]) + arr[buffer_idx, self.s2i['$$x_c$$']]
        out_pyr = np.array([[np.sum(pyr_summands[:n_], axis=0) for n_ in self.n_range]]) 
        return out_pxr, out_pyr

    def __pxm_pym_features(self, arr : np.array, buffer_idx : int):
        # Function will only work for step-wise calculations, ie arr[buffer_idx, ...]
        l_list = [self.constants['str_to_val'][f"l_{n_}"] for n_ in self.n_range]
        r_list = [self.constants['str_to_val'][f"r_{n_}"] for n_ in self.n_range]
        pxm_summands = [-l_list[n_-1]*np.sin(arr[buffer_idx, self.s2i[fr"$$\theta_{{{n_}}}$$"]]) for n_ in self.n_range] 
        pym_summands = [ l_list[n_-1]*np.cos(arr[buffer_idx, self.s2i[fr"$$\theta_{{{n_}}}$$"]]) for n_ in self.n_range] 
        out_pxm = np.array([[np.sum(pxm_summands[:n_-1], axis=0) - r_list[n_-1]*np.sin(arr[buffer_idx, self.s2i[fr"$$\theta_{{{n_}}}$$"]]) for n_ in self.n_range]]) + arr[buffer_idx, self.s2i['$$x_c$$']]
        out_pym = np.array([[np.sum(pym_summands[:n_-1], axis=0) + r_list[n_-1]*np.cos(arr[buffer_idx, self.s2i[fr"$$\theta_{{{n_}}}$$"]]) for n_ in self.n_range]]) 
        return out_pxm, out_pym    
            
    def __pxm_pym_derivatives(self, arr : np.array, buffer_idx : int): # is a support feature function for energy features (only used there)
        # Function will only work for step-wise calculations, ie arr[buffer_idx, ...]
        l_list = np.array([self.constants['str_to_val'][f"l_{n_}"] for n_ in self.n_range])
        r_list = np.array([self.constants['str_to_val'][f"r_{n_}"] for n_ in self.n_range])
        diff_pxm_summands = [-l_list[n_-1] * arr[buffer_idx, self.s2i[fr'$$\dot{{\theta_{n_}}}$$']] * np.cos(arr[buffer_idx, self.s2i[fr"$$\theta_{{{n_}}}$$"]]) for n_ in self.n_range]
        diff_pym_summands = [-l_list[n_-1] * arr[buffer_idx, self.s2i[fr'$$\dot{{\theta_{n_}}}$$']] * np.sin(arr[buffer_idx, self.s2i[fr"$$\theta_{{{n_}}}$$"]]) for n_ in self.n_range] 
        out_dpxm = np.array([[np.sum(diff_pxm_summands[:n_-1], axis=0) - r_list[n_-1] * arr[buffer_idx, self.s2i[fr'$$\dot{{\theta_{n_}}}$$']] * np.cos(arr[buffer_idx, self.s2i[fr"$$\theta_{{{n_}}}$$"]]) for n_ in self.n_range]]) + arr[buffer_idx, self.s2i['$$\dot{x_c}$$']]
        out_dpym = np.array([[np.sum(diff_pym_summands[:n_-1], axis=0) - r_list[n_-1] * arr[buffer_idx, self.s2i[fr'$$\dot{{\theta_{n_}}}$$']] * np.sin(arr[buffer_idx, self.s2i[fr"$$\theta_{{{n_}}}$$"]]) for n_ in self.n_range]])
        return out_dpxm, out_dpym
    
    ### Regular Features
    def _pxr_pyr_features(self, arr : np.array, buffer_idx : int):
        out_pxr, out_pyr = self.__pxr_pyr_features(arr, buffer_idx)
        for n_ in self.n_range: 
            arr[buffer_idx, self.s2i[f"$$p_{{x_{n_}}}^{{r}}$$"]] = out_pxr[-1, n_-1]
            arr[buffer_idx, self.s2i[f"$$p_{{y_{n_}}}^{{r}}$$"]] = out_pyr[-1, n_-1]       

    def _pxm_pym_features(self, arr : np.array, buffer_idx : int):
        out_pxm, out_pym = self.__pxm_pym_features(arr, buffer_idx)
        for n_ in self.n_range: 
            arr[buffer_idx, self.s2i[f"$$p_{{x_{n_}}}^{{m}}$$"]] = out_pxm[-1, n_-1]
            arr[buffer_idx, self.s2i[f"$$p_{{y_{n_}}}^{{m}}$$"]] = out_pym[-1, n_-1]
    
    def _sin_thetas(self, arr : np.array, buffer_idx : int):
        for n_ in self.n_range:
            arr[buffer_idx, self.s2i[fr"$$\sin(\theta_{{{n_}}})$$"]] = np.sin(arr[buffer_idx, self.s2i[fr"$$\theta_{{{n_}}}$$"]])

    def _cos_thetas(self, arr : np.array, buffer_idx : int):
        for n_ in self.n_range:
            arr[buffer_idx, self.s2i[fr"$$\cos(\theta_{{{n_}}})$$"]] = np.cos(arr[buffer_idx, self.s2i[fr"$$\theta_{{{n_}}}$$"]])

    def _alphas(self, arr : np.array, buffer_idx : int):  # 1/2*n*(n-1) features
        tuples = self.fgroup2tup["Rod-to-Rod Angles"] # --> [(1, 2), (2, 3), (3, 4), (1, 3), (2, 4), (1, 4)] for n=4
        for tup in tuples:
            arr[buffer_idx, self.s2i[fr"$$\alpha_{{{tup[0]},{tup[1]}}}$$"]] = arr[buffer_idx, self.s2i[fr"$$\theta_{{{tup[1]}}}$$"]] - \
                                                                              arr[buffer_idx, self.s2i[fr"$$\theta_{{{tup[0]}}}$$"]] 

    def _sin_alphas(self, arr : np.array, buffer_idx : int):  # 1/2*n*(n-1) features
        tuples = self.fgroup2tup["Rod-to-Rod Angles"] # --> [(1, 2), (2, 3), (3, 4), (1, 3), (2, 4), (1, 4)] for n=4
        for tup in tuples:
            arr[buffer_idx, self.s2i[fr"$$\sin(\alpha_{{{tup[0]},{tup[1]}}})$$"]] = np.sin(arr[buffer_idx, self.s2i[fr"$$\theta_{{{tup[1]}}}$$"]] - \
                                                                                   arr[buffer_idx, self.s2i[fr"$$\theta_{{{tup[0]}}}$$"]] )

    def _cos_alphas(self, arr : np.array, buffer_idx : int):  # 1/2*n*(n-1) features       
        tuples = self.fgroup2tup["Rod-to-Rod Angles"] # --> [(1, 2), (2, 3), (3, 4), (1, 3), (2, 4), (1, 4)] for n=4
        for tup in tuples:
            arr[buffer_idx, self.s2i[fr"$$\cos(\alpha_{{{tup[0]},{tup[1]}}})$$"]] = np.cos(arr[buffer_idx, self.s2i[fr"$$\theta_{{{tup[1]}}}$$"]] - \
                                                                                   arr[buffer_idx, self.s2i[fr"$$\theta_{{{tup[0]}}}$$"]] )

    def _avg_rod_tip_features(self, arr : np.array, buffer_idx : int):
        out_pxr, out_pyr = self.__pxr_pyr_features(arr, buffer_idx)
        arr[buffer_idx, self.s2i[r"$$\bar{p^r_x}$$"]]        = np.sum(out_pxr)/self.n
        arr[buffer_idx, self.s2i[r"$$\bar{p^r_y}$$"]]        = np.sum(out_pyr)/self.n
        arr[buffer_idx, self.s2i[r"$$\bar{\theta_{r-c}}$$"]] = np.sum([np.arctan2(out_pyr[-1, n_-1], out_pxr[-1, n_-1]) for n_ in self.n_range])/self.n
        arr[buffer_idx, self.s2i[r"$$\bar{p^d_{r-c, x}}$$"]] = (np.sum(out_pxr) - arr[buffer_idx, self.s2i[f"$$x_c$$"]])/self.n
        arr[buffer_idx, self.s2i[r"$$\bar{d_{r-c}}$$"]]      = np.sum([np.sqrt((out_pxr[-1, n_-1] - arr[buffer_idx, self.s2i['$$x_c$$']])**2 + out_pyr[-1, n_-1]**2) for n_ in self.n_range])/self.n

    def _avg_com_features(self, arr : np.array, buffer_idx : int):
        m_list = np.array([self.constants['str_to_val'][f"m_{n_}"] for n_ in self.n_range])
        m_total = np.sum(m_list)
        out_pxm, out_pym = self.__pxm_pym_features(arr, buffer_idx)
        arr[buffer_idx, self.s2i[r"$$\bar{p^m_x}$$"]]         = np.sum(out_pxm)/self.n # m: mass-independent
        arr[buffer_idx, self.s2i[r"$$\bar{p^m_y}$$"]]         = np.sum(out_pym)/self.n
        arr[buffer_idx, self.s2i[r"$$\bar{p^{m*}_x}$$"]]      = np.dot(out_pxm, m_list)/(self.n*m_total) # m*: mass-dependent
        arr[buffer_idx, self.s2i[r"$$\bar{p^{m*}_y}$$"]]      = np.dot(out_pym, m_list)/(self.n*m_total)
        arr[buffer_idx, self.s2i[r"$$\bar{\theta_{m-c}}$$"]]  = np.arctan2(np.sum(out_pym)/self.n, np.sum(out_pxm)/self.n)
        arr[buffer_idx, self.s2i[r"$$\bar{\theta_{m*-c}}$$"]] = np.arctan2(np.dot(out_pym, m_list)/(self.n*m_total), np.dot(out_pxm, m_list)/(self.n*m_total))
        arr[buffer_idx, self.s2i[r"$$\bar{p^d_{m-c, x}}$$"]]  = (np.sum(out_pxm) - arr[buffer_idx, self.s2i[f"$$x_c$$"]])/self.n
        arr[buffer_idx, self.s2i[r"$$\bar{p^d_{m*-c, x}}$$"]] = (np.dot(out_pxm, m_list) - arr[buffer_idx, self.s2i[f"$$x_c$$"]])/(self.n*m_total)
        arr[buffer_idx, self.s2i[r"$$\bar{d_{m-c}}$$"]]       = np.sum(np.sqrt([  (out_pxm[-1, n_-1] - arr[buffer_idx, self.s2i['$$x_c$$']])**2                       + (out_pym[-1, n_-1])**2 
                                                                                    for n_ in self.n_range]))/self.n
        arr[buffer_idx, self.s2i[r"$$\bar{d_{m*-c}}$$"]]      = np.sum(np.sqrt([ ((out_pxm[-1, n_-1] - arr[buffer_idx, self.s2i['$$x_c$$']])*m_list[n_-1]/m_total)**2 + (out_pym[-1, n_-1]*m_list[n_-1]/m_total)**2 
                                                                                    for n_ in self.n_range]))/self.n

    def _rod_to_rod_polar_distances(self, arr : np.array, buffer_idx : int):
        out_pxr, out_pyr = self.__pxr_pyr_features(arr, buffer_idx)
        tuples = self.fgroup2tup["Rod-Tip-to-Rod-Tip (Polar) Distances"] 
        # --> [(0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (2, 4)] for n=4; polar r-r distance; n(n-1)/2 features
        for tup_ in tuples:
            if tup_[0] == 0:
                arr[buffer_idx, self.s2i[f"$$d^{{r-r}}_{{{tup_[0]}, {tup_[1]}}}$$"]] = np.sqrt( (arr[buffer_idx, self.s2i[f"$$x_c$$"]] - out_pxr[-1, tup_[1]-1])**2 + \
                                                                                                (0 - out_pyr[-1, tup_[1]-1])**2 )                
            else:
                arr[buffer_idx, self.s2i[f"$$d^{{r-r}}_{{{tup_[0]}, {tup_[1]}}}$$"]] = np.sqrt( (out_pxr[-1, tup_[0]-1] - out_pxr[-1, tup_[1]-1])**2 + \
                                                                                                (out_pyr[-1, tup_[0]-1] - out_pyr[-1, tup_[1]-1])**2 )

    def _com_to_com_polar_distances(self, arr : np.array, buffer_idx : int):
        out_pxm, out_pym = self.__pxm_pym_features(arr, buffer_idx)
        tuples = self.fgroup2tup["Center-of-Mass-to-Center-of-Mass (Polar) Distances"]  
        # --> [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)] for n=4; polar m-m distances; n(n-1)/2 features
        for tup_ in tuples:
            arr[buffer_idx, self.s2i[f"$$d^{{m-m}}_{{{tup_[0]}, {tup_[1]}}}$$"]] = np.sqrt( (out_pxm[-1, tup_[0]-1] - out_pxm[-1, tup_[1]-1])**2 + \
                                                                                    (out_pym[-1, tup_[0]-1] - out_pym[-1, tup_[1]-1])**2 )      

    def _rod_to_com_polar_distances(self, arr : np.array, buffer_idx : int):
        out_pxr, out_pyr = self.__pxr_pyr_features(arr, buffer_idx)
        out_pxm, out_pym = self.__pxm_pym_features(arr, buffer_idx)
        tuples = self.fgroup2tup["Rod-Tip-to-Center-of-Mass (Polar) Distances"]
        # --> [(0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (2, 1), (2, 4), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3)] for n=4; polar r-m distances; (n-1)(n-2)+2(n-1) features
        for tup_ in tuples:
            if tup_[0] == 0:
                arr[buffer_idx, self.s2i[f"$$d^{{r-m}}_{{{tup_[0]}, {tup_[1]}}}$$"]] = np.sqrt( (arr[buffer_idx, self.s2i[f"$$x_c$$"]] - out_pxm[-1, tup_[1]-1])**2 + \
                                                                                                (0 - out_pym[-1, tup_[1]-1])**2 )               
            else:
                arr[buffer_idx, self.s2i[f"$$d^{{r-m}}_{{{tup_[0]}, {tup_[1]}}}$$"]] = np.sqrt( (out_pxr[-1, tup_[0]-1] - out_pxm[-1, tup_[1]-1])**2 + \
                                                                                                (out_pyr[-1, tup_[0]-1] - out_pym[-1, tup_[1]-1])**2 )

    def _rod_to_rod_cartesian_distances(self, arr : np.array, buffer_idx : int):
        out_pxr, out_pyr = self.__pxr_pyr_features(arr, buffer_idx)
        tuples = self.fgroup2tup["Rod-Tip-to-Rod-Tip (Cartesian) Distances"]
        # --> [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] for n=3; cartesian r-r distances; 2 * n(n-1) features
        for tup_ in tuples:
            if tup_[0] == 0:
                arr[buffer_idx, self.s2i[f"$$d^{{r-r}}_{{x_{{{tup_[0]}, {tup_[1]}}}}}$$"]] = arr[buffer_idx, self.s2i[f"$$x_c$$"]] - out_pxr[-1, tup_[1]-1]
                # arr[buffer_idx, self.s2i[f"$$d^{{r-r}}_{{y_{{{tup_[0]}, {tup_[1]}}}}}$$"]] = 0 - out_pyr[-1, tup_[1]-1] # this would be one redundant feature
            else:
                arr[buffer_idx, self.s2i[f"$$d^{{r-r}}_{{x_{{{tup_[0]}, {tup_[1]}}}}}$$"]] = out_pxr[-1, tup_[0]-1] - out_pxr[-1, tup_[1]-1]
                arr[buffer_idx, self.s2i[f"$$d^{{r-r}}_{{y_{{{tup_[0]}, {tup_[1]}}}}}$$"]] = out_pyr[-1, tup_[0]-1] - out_pyr[-1, tup_[1]-1]

    def _com_to_com_cartesian_distances(self, arr : np.array, buffer_idx : int):
        out_pxm, out_pym = self.__pxm_pym_features(arr, buffer_idx)
        tuples = self.fgroup2tup["Center-of-Mass-to-Center-of-Mass (Cartesian) Distances"]
        # --> [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)] for n=4; cartesian m-m distances; 2 * n(n-1)/2 features
        for tup_ in tuples:
            arr[buffer_idx, self.s2i[f"$$d^{{m-m}}_{{x_{{{tup_[0]}, {tup_[1]}}}}}$$"]] = out_pxm[-1, tup_[0]-1] - out_pxm[-1, tup_[1]-1]
            arr[buffer_idx, self.s2i[f"$$d^{{m-m}}_{{y_{{{tup_[0]}, {tup_[1]}}}}}$$"]] = out_pym[-1, tup_[0]-1] - out_pym[-1, tup_[1]-1]

    def _rod_to_com_cartesian_distances(self, arr : np.array, buffer_idx : int):
        out_pxr, out_pyr = self.__pxr_pyr_features(arr, buffer_idx)
        out_pxm, out_pym = self.__pxm_pym_features(arr, buffer_idx)
        tuples = self.fgroup2tup["Rod-Tip-to-Center-of-Mass (Cartesian) Distances"]
        # --> [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)] for n=3; cartesian r-m distances; 2 * n(n+1)
        for tup_ in tuples:
            if tup_[0] == 0:
                arr[buffer_idx, self.s2i[f"$$d^{{r-m}}_{{x_{{{tup_[0]}, {tup_[1]}}}}}$$"]] = arr[buffer_idx, self.s2i[f"$$x_c$$"]] - out_pxm[-1, tup_[1]-1]
                #arr[buffer_idx, self.s2i[f"$$d^{{r-m}}_{{y_{{{tup_[0]}, {tup_[1]}}}}}$$"]] = 0 - out_pym[-1, tup_[1]-1] # this would be one redundant feature 
            else:
                arr[buffer_idx, self.s2i[f"$$d^{{r-m}}_{{x_{{{tup_[0]}, {tup_[1]}}}}}$$"]] = out_pxr[-1, tup_[0]-1] - out_pxm[-1, tup_[1]-1]
                arr[buffer_idx, self.s2i[f"$$d^{{r-m}}_{{y_{{{tup_[0]}, {tup_[1]}}}}}$$"]] = out_pyr[-1, tup_[0]-1] - out_pym[-1, tup_[1]-1]    
        
    def _ernergy_features(self, arr : np.array, buffer_idx : int):
        g = self.constants['str_to_val']['g']
        m_list = [self.constants['str_to_val'][f"m_{n_}"] for n_ in self.n_range]
        m_c = self.constants['str_to_val']["m_c"]
        I_list = [self.constants['str_to_val'][f"I_{n_}"] for n_ in self.n_range]
        out_dpxm, out_dpym = self.__pxm_pym_derivatives(arr, buffer_idx)
        for n_ in self.n_range:
            arr[buffer_idx, self.s2i[fr'$$V_{{{n_}}}$$']] = m_list[n_-1]*g*arr[buffer_idx, self.s2i[f"$$p_{{y_{n_}}}^{{m}}$$"]] 
        arr[buffer_idx, self.s2i[fr'$$T_{{trans_{{c}}}}$$']] = 0.5*m_c*arr[buffer_idx, self.s2i['$$\dot{x_c}$$']]**2
        for n_ in self.n_range: 
            arr[buffer_idx, self.s2i[fr'$$T_{{trans_{{{n_}}}}}$$']] = 0.5*m_list[n_-1]*(out_dpxm[-1, n_-1]**2 + out_dpym[-1, n_-1]**2) 
        for n_ in self.n_range:    
            arr[buffer_idx, self.s2i[fr'$$T_{{rot_{n_}}}$$']] = 0.5*I_list[n_-1]*arr[buffer_idx, self.s2i[fr'$$\dot{{\theta_{n_}}}$$']]**2
        arr[buffer_idx, self.s2i[fr'$$T_{{trans}}$$']] = arr[buffer_idx, self.s2i[fr'$$T_{{trans_{{c}}}}$$']] + np.sum([arr[buffer_idx, self.s2i[fr'$$T_{{trans_{{{n_}}}}}$$']] for n_ in self.n_range], axis=0) 
        arr[buffer_idx, self.s2i[fr'$$T_{{rot}}$$']]   = np.sum([arr[buffer_idx, self.s2i[fr'$$T_{{rot_{n_}}}$$']] for n_ in self.n_range], axis=0) 
        arr[buffer_idx, self.s2i[fr'$$V$$']]           = np.sum([arr[buffer_idx, self.s2i[fr'$$V_{{{n_}}}$$']]     for n_ in self.n_range], axis=0) 
        arr[buffer_idx, self.s2i[fr'$$T$$']]           = arr[buffer_idx, self.s2i[fr'$$T_{{trans}}$$']] + arr[buffer_idx, self.s2i[fr'$$T_{{rot}}$$']]
        arr[buffer_idx, self.s2i[fr'$$L$$']]           = arr[buffer_idx, self.s2i[fr'$$T$$']] - arr[buffer_idx, self.s2i[fr'$$V$$']]
        