import pandas
import numpy as np
import os.path

train_result_files = {
    -1: 'y_train_smpl.csv',
    0: 'y_train_smpl_0.csv',
    1: 'y_train_smpl_1.csv',
    2: 'y_train_smpl_2.csv',
    3: 'y_train_smpl_3.csv',
    4: 'y_train_smpl_4.csv',
    5: 'y_train_smpl_5.csv',
    6: 'y_train_smpl_6.csv',
    7: 'y_train_smpl_7.csv',
    8: 'y_train_smpl_8.csv',
    9: 'y_train_smpl_9.csv'
}

test_result_files = {
    -1: 'y_test_smpl.csv',
    0: 'y_test_smpl_0.csv',
    1: 'y_test_smpl_1.csv',
    2: 'y_test_smpl_2.csv',
    3: 'y_test_smpl_3.csv',
    4: 'y_test_smpl_4.csv',
    5: 'y_test_smpl_5.csv',
    6: 'y_test_smpl_6.csv',
    7: 'y_test_smpl_7.csv',
    8: 'y_test_smpl_8.csv',
    9: 'y_test_smpl_9.csv'
}


def result_file_selector(id, train=True):
    """Get the result filenames using an integer id

    :param id: integer result labels id
    :type id: int
    :param train: Train or test data, defaults to True
    :type train: bool, optional
    :return: identifier for datafile
    :rtype: str
    """
    if train:
        return train_result_files.get(id, 'y_train_smpl.csv')
    else :
        return test_result_files.get(id, 'y_train_smpl.csv')


def get_file_path(filename, my_path=os.path.abspath(os.path.dirname("Data/"))):
    """Construct the filepath string, takes name of file as arg

    :param filename: The name of the file to be imported
    :type filename: string
    :return: The absolute path to the file
    :rtype: string
    """
    return os.path.join(my_path, filename)

def get_dataset(filepath):
    """Use get_random_data() instead. 
    Get Dataframe with randomized instance order, takes filepath as arg

    :param filepath: The full absolute path to the file
    :type filepath: string
    :return: The entire data collection with randomized order of instances
    :rtype: pandas datafram
    """
    return pandas.read_csv(filepath, header='infer')



def get_data(result_id=-1, train=True):
    """Get a tuple of the result and data csv

    :param result_id: The index of the result datafile, defaults to -1
    :type result_id: int, optional
    :return: A tuple of the data collection
    :rtype: (pandas.df, pandas.df)
    """
    if train:
        x = get_dataset(get_file_path('x_train_gr_smpl.csv'))
    else:
        x = get_dataset(get_file_path('x_test_gr_smpl.csv'))
    filePicker = result_file_selector(result_id, train)
    y = get_dataset(get_file_path(filePicker))
    y.columns = ['y']
    return x, y


def append_result_col(data, result):
    """Return a new dataframe with result column in data

    :param data: The dataset without the result column
    :type data: pandas.df
    :param result: The result vector to append to data
    :type result: pandas.df
    """
    result.columns = ['y']
    return data.join(result)


def balance_by_class(X, y, max_data_size=None, allow_imbalanced_class_dist=False, random_state=0):
    """Select a sample of the data with a balanced class distribution

    :param X: data
    :type X: pandas.df
    :param y: labels
    :type y: pandas.df
    :param max_data_size: size of sample. Defaults to None -> in this case the sample returned will be the size of the smallest class
    :type max_data_size: int, Optional
    :param allow_imbalanced_class_dist: If size param > number of rows in smallest class indicate if allowing an imbalanced class distribution is ok
    :type allow_imbalanced_class_dist: Bool, Optional
    :param random_state: Random_state used for ksplits and balancing, defaults to 0
    :type random_state: int, optional
    :return: the sample and labels
    :rtype: tuple(pandas.df, pandas.df)
    """
    data = append_result_col(X, y)
    classes = np.unique(y)
    datasets = []
    smallest_size = data.shape[0]
    for i in range(len(classes)):
        temp_sample = data[data['y'] == classes[i]]
        datasets += [temp_sample.sample(frac=1, random_state=random_state)]
        if temp_sample.shape[0] < smallest_size:
            smallest_size = temp_sample.shape[0]
    frame = pandas.DataFrame(columns=data.columns)
    if max_data_size is None:
        for df in datasets:
            frame = frame.append(df.head(smallest_size))
    else:
        if allow_imbalanced_class_dist and max_data_size > smallest_size:
            raise ValueError(
                "Size argument is too large for a balanced dataset")
        for df in datasets:
            frame = frame.append(df.head(max_data_size))
    y_res = frame[['y']]
    X_res = frame.drop('y', 1)
    return X_res, y_res
