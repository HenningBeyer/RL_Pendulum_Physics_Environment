import os 

def find_new_numeric_dir_name(base_dir, dir_prefix):
    """ Finds a new dir name inside base_exp_dir for a specified dir_prefix, to later create a new dir, etc..
        Example:  - base_exp_dir = os.getcwd(), dir_prefix = 'example_experiment_'
                  - There is just one dir in os.getcwd() with the specified prefix: 'example_experiment_1'
                  - so this function returns 'example_experiment_2'
    """
    dirs = os.listdir(base_dir) # lists all files and dirs in a single directory (only depth 1)
    dir_prefixes = [dir_.replace(dir_prefix, '') for dir_ in dirs if dir_prefix in dir_] # 'example_experiment_1' --> 1
    max_prefix_num = max([int(prefix_) for prefix_ in dir_prefixes if prefix_.isnumeric()] + [0]) # ['a1', 'x', 'test', '1'] --> max([1, 0]) --> 1
    return dir_prefix + str(max_prefix_num+1) # --> 'example_experiment_2'

def find_new_numeric_file_name(base_dir, file_prefix, file_extension):
    """ Finds a new file name inside base_exp_dir for a specified file_prefix, to later create a new dir, etc..
        Example:  - base_dir = os.getcwd() + '\\experiment_1', file_prefix = 'sample_dataset_', file_extension = '.csv'
                  - There is just one file in os.getcwd() with the specified prefix: 'sample_dataset_1.csv'
                  - so this function returns 'sample_dataset_2.csv'
    """
    dirs = os.listdir(base_dir) # lists all files and dirs in a single directory (only depth 1)
    file_prefixes  = [file_.replace(file_prefix, '') for file_ in dirs if file_prefix in file_] # 'sample_dataset_.csv' --> '1.csv'
    name_prefixes  = [prefix_.replace(file_extension, '') for prefix_ in file_prefixes if file_extension in prefix_] # '1.csv' --> '1'
    max_prefix_num = max([int(prefix_) for prefix_ in name_prefixes if prefix_.isnumeric()] + [0]) # ['a1', 'x', 'test', '1'] --> max([1, 0]) --> 1
    return file_prefix + str(max_prefix_num+1) + file_extension # --> 'sample_dataset_2.csv'