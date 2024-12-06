import numpy as np

class FE_Replay_Buffer:
    """ A replay buffer explicitly made for feature engineering
            --> the buffer is implemented in numpy, as this provides a lot faster feature engineering capabilities (torch is slow in basic operations and indexing)
            --> also the fe will put everything (states, actions, rewards, ...) in one buffer with one assignment (also faster than the RL replay buffer)
                --> the same state vector that is assigned here, is also split into (states, actions, rewards, ...) data package
                    and passed to the RL replay buffer, where they are converted to a torch.tensor
            --> the state will be sampled from the latest FE output vector and passed to the RL replay buffer

            - The time for cpu-to

        Note: It is still recommended to use an Episode_
            --> for any continuous or robotic tasks, this additional memory often smaller than 1 MB, and thus no problem.
            --> it would be possible to trim the size of this FE replay buffer to the maximum feature length actually needed for FE
                - but this is not done for simplicity
                --> actually, this replay buffer is way easier to use for feature visualization than the RL replay buffer!

        The replay buffer data is not needed for any optimization outside the environment (for this case with PPO).

        Source: https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl/train/replay_buffer.py
        all PER-related code is removed. everything (state, action, reward) is stored inside the buffer
        --> the agent will get these features for feature engineering
    """


    def __init__(self,
                 max_size: int, # specify the max size in steps, this speeds up adding new states, as memory is pre-occupied
                 num_feats):
        self.p = 0  # pointer
        self.if_full = False
        self.cur_size = 0
        self.add_size = 0 # number of states to add
        self.max_size = max_size
        self.num_feats = num_feats
        self.buffer_is_static = False

        """ The buffer will be a np.array as feature engineering is a lot faster with np.arrays.
            torch.tensors are used in most other to store replay-buffer data on GPUs (not possible with np.array)
            to reduce cpu-to-gpu data transfers, but this is not needed for these tasks:
                - The states are small (at max 300 float numbers per pass --> transfer takes around 0.1 ms on to a common single GPU
                --> so the used concept of GPU acceleration of agents with cpu simulation still performs well without problems
                --> also: custom environments are easier to accelerate with cpu, especially when prototyping

            During feature engineering, the replay buffer should be updated by reference for optimal speed and simplicity.
                --> pass the buffer.buffer reference with the last index, and assign new features directly to buffer.buffer
        """
        """ Currently tested only for 2-D arrays, and not 3-D arrays (batched assignment)
        """
        self.buffer = np.empty((max_size, num_feats), dtype=np.float32) # pre-allocate all memory needed

    def update(self, states : np.array):
        """ A very efficient update function, that does not allocate additional memory.
            An update will only take at most 1ms and is very fast!

            states can be like np.array([[1,2,3,4], [2,3,4,5], ...]) (a 2-D array of arbitrary number of states)
        """
        if self.buffer_is_static == True:
            # This allows to iterate through the environment without updating anything
            # --> Just used for speed benchmarking
            return

        # states can be np.array([[1,2,3,4], [2,3,4,5], ...]) a 2-D array of arbitrary number of states
        self.add_size = states.shape[0]
        p = self.p + self.add_size  # pointer
        if p <= self.max_size: # simply assign rows for a buffer with space left
            self.buffer[self.p:p] = states
        else:
            # This else-statement is once called to assign all overflowing states at the beginning
            # Then, the 'pointer' p is set, so that the if statment one is called until the next overflowing,
            #   --> the buffer fills up starting from the bottom again

            # raise Exception('The replay buffer is out of size! This error is raised to prevent data to be overwritten under the hood.')
            # (for production, you might want to remove this Exception)
            # (for testing, may enable it)

            self.if_full = True
            p0 = self.p
            p1 = self.max_size
            p2 = self.max_size - self.p
            p = p - self.max_size

            self.buffer[p0:p1], self.buffer[0:p] = states[:p2], states[-p:]

        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p

    def sample_last_state(self):
        """ Get the last state from the buffer """
        return self.buffer[self.p-1]

    def get_last_state_idx(self):
        """ This is rather used for feature engineering to start slicing at self.p-1 """
        return self.p