import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture as GMM

""" Table header:
col1:  filename
col2:  duration in msec
col3:  f0 at "steady state"
col4:  F1 at "steady state"
col5:  F2 at "steady state"
col6:  F3 at "steady state"
col7:  F4 at "steady state"
col8:  F1 at 20% of vowel duration
col9:  F2 at 20% of vowel duration
col10: F3 at 20% of vowel duration
col11: F1 at 50% of vowel duration
col12: F2 at 50% of vowel duration
col13: F3 at 50% of vowel duration
col14: F1 at 80% of vowel duration
col15: F2 at 80% of vowel duration
col16: F3 at 80% of vowel duration
"""
vowels = ["ae", "ah", "aw", "eh", "er",
          "ei", "ih", "iy", "oa", "oo", "uh", "uw"]

header = ["duration", "f0s", "F1s", "F2s", "F3s", "F4s", "F1_20", "F2_20", "F3_20",
          "F1_50", "F2_50", "F3_50", "F1_80", "F2_80", "F3_80", "person", "sample", "vowel"]

def load_data():
    """Loads and reshapes the data putting it into a pandas dataframe
    
    Returns:
        [pd.DataFrame]
    """
    df = pd.read_csv("Wovels/vowdata_nohead.dat",
                     delim_whitespace=True, header=None)
    first_column = df[0].values
    person_list = pd.Series([entry[:1] for entry in first_column])
    sample_list = pd.Series([entry[1:3] for entry in first_column])
    vowel_list = pd.Series([entry[3:] for entry in first_column])
    del df[0]
    df = pd.concat([df, person_list, sample_list,
                    vowel_list], axis=1, sort=False)
    df.columns = header
    df = df.set_index("vowel")

    # Standardization 
    for feature in header[:-3]:
        print(feature)
        mean = df[feature].mean()
        std = df[feature].std()
        df[feature] = (df[feature] - mean)/(std)

    # Normalize min-max
    for feature in header[:-3]:
        minimum = df[feature].min()
        maximum = df[feature].max()
        df[feature] = (df[feature] - minimum)/(maximum-minimum)


    

    df = df.drop(["duration", "f0s", "F1s", "F2s", "F3s", "F4s"],axis=1)
    print(df.head)
    return df


def split_data(data, train_samples):
    """Splits the data into training and test sets,|
        specify the number of training samples, the rest of the samples
        are used for testing

    
    Arguments:
        data {[pd.DataFrame]} -- [description]
        train_samples {[int]} -- [The number of training sampless]
    
    Returns:
        [pd.DataFrame] -- [training dataframe]
        [pd.DataFrame] -- [test dataframe]
    """
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    for vowel in vowels:
        train_df = train_df.append(df.loc[vowel][:train_samples])
        test_df = test_df.append(df.loc[vowel][train_samples:])
    return train_df, test_df


def get_mean(data):
    """Gets the mean value in each column for each
        vowel in the dataframe. 
    
    Arguments:
        data {[pd.DataFrame]} -- [Dataframe with labeled index]
    
    Returns:
        [pd.DataFrame] -- [Dataframe with mean value in columns for each vowel]
    """
    mean_df = pd.DataFrame()

    for vowel in vowels:
        vowel_df = data.loc[vowel]
        vowel_df = vowel_df.iloc[:, :-2]
        vowel_df = vowel_df.mean(axis=0)
        mean_df[vowel] = vowel_df
    mean_df = mean_df.set_index(vowel_df.index)

    return mean_df


def get_covariance_matrix(data, diagonal=False):
    """[summary]
    
    Arguments:
        data {[type]} -- [description]
    
    Keyword Arguments:
        diagonal {bool} -- [description] (default: {False})
    
    Returns:
        [type] -- [description]
    """
    cov_matrix_dict = {}
    for vowel in vowels:
        vowel_df = data.loc[vowel]
        vowel_df = vowel_df.iloc[:, :-2]
        vowel_df = vowel_df.cov()
        cov_matrix_dict[vowel] = vowel_df
    return cov_matrix_dict


def train_single_GM(train_data, diagonal=False):
    rv_list = []
    cov_dict = get_covariance_matrix(train_data)
    mean_matrix = get_mean(train_data)
    for vowel in vowels:
        cov_matrix = cov_dict[vowel].values.astype(float)
        mean = mean_matrix[vowel].values.astype(float)

        # Multivarate normal distribution
        rv = multivariate_normal(mean=mean, cov=cov_matrix)
        rv_list.append(rv)

    # No need to do scaling since all classes have same number of samples???
    return rv_list


def test_singel_GMM(rv_list, test_data):
    """[summary]
    
    Arguments:
        rv_list {[type]} -- [description]
        test_data {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    data_length = test_data.shape[0]
    probabilities = np.zeros((len(vowels), data_length))
    data_values = test_data.values[:, :-2].astype(int)
    actual = test_data.index.tolist()

    for index in range(len(vowels)):
        vowel_rv = rv_list[index]
        probabilities[index] = vowel_rv.pdf(data_values)

    predictions = np.argmax(probabilities, axis=0)

    return predictions, actual


def train_GMM(train_data, n_components):
    """[summary]
    
    Arguments:
        train_data {[type]} -- [description]
        n_components {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    gmm_list = []
    for vowel in vowels:
        training_values = train_data.loc[vowel].values[:, :-2]
        # Create a GMM
        gmm = GMM(n_components=n_components, covariance_type='diag',
                  reg_covar=1e-4, random_state=0)

        # Train the GMM on the training data
        gmm.fit(training_values)  # What are the labels?
        # Add the GMM to the list of all GMMs, one for each class
        gmm_list.append(gmm)

    return gmm_list


def test_multiple_GMM(gmm_list, test_data, n_components):
    """[summary]
    
    Arguments:
        gmm_list {[type]} -- [description]
        test_data {[type]} -- [description]
        n_components {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    # Initialize empty arrays
    probabilities = np.zeros((len(vowels), test_data.shape[0]))
    data_values = test_data.values[:, :-2]
    actual = test_data.index.tolist()
    for index, vowel in enumerate(vowels):

        gmm = gmm_list[index]
        # Find the total predicted probability over all the components of the mixture model
        for j in range(n_components):
            N = multivariate_normal(mean=gmm.means_[j], cov=gmm.covariances_[j], allow_singular=True)
            probabilities[index] += gmm.weights_[j] * N.pdf(data_values)
        # This is equivalent to: testing_preds[i] = gmm.score_samples(x_test)

    # The prediction is the class that had the highest probability
    predictions = np.argmax(probabilities, axis=0)
    return predictions, actual


def get_confusion_matrix(predictions, labels):
    """[summary]
    
    Arguments:
        predictions {[type]} -- [description]
        labels {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    data_length = predictions.shape[0]
    conf_matrix = np.zeros((len(vowels), len(vowels)))
    for i in range(data_length):
        vowel_index = vowels.index(labels[i])
        conf_matrix[vowel_index][(predictions[i])] += 1

    return conf_matrix


def get_error_rate(conf_matrix):
    """[summary]
    
    Arguments:
        conf_matrix {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    error_rate = (1 - np.sum(conf_matrix.diagonal())/np.sum(conf_matrix))*100
    return error_rate


if __name__ == "__main__":

    # Loading the data and split into train and test sets
    df = load_data()
    train, test = split_data(df, 69)

    # Part 1: using a single gaussian to predict classes
    rvs = train_single_GM(train)
    preds, labels = test_singel_GMM(rvs, test)
    conf = get_confusion_matrix(preds, labels)
    error_rate = get_error_rate(conf)
    print("Confusion matrix:")
    print(conf)
    print(f"Error rate: {error_rate}%")

    # Part 2: Using a mixture of 3 gaussians per class to predict the class
    gmm_list = train_GMM(train,3)
    preds, labels = test_multiple_GMM(gmm_list, test, 3)
    conf = get_confusion_matrix(preds, labels)
    error_rate = get_error_rate(conf)
    print("Confusion matrix:")
    print(conf)
    print(f"Error rate: {error_rate}%")