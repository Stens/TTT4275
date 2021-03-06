import numpy as np
import matplotlib.pyplot as plt
import copy
import seaborn as sn
import pandas as pd

CLASSES = 3
legends = ['Setosa', 'Versicolour', 'Virginica']

def load_data():
    """Loads the iris dataset from file and 
    
    Returns:
        np.array -- Numpy array with features in the first four columns, 
                       then a column of ones followed by a column of labels
    """
    for i in range(CLASSES):
        tmp = np.loadtxt("./Iris_TTT4275/class_"+str(i+1),delimiter=",")
       
        # Add the class, and 1
        class_number = np.ones((tmp.shape[0],2)) 
        class_number[:,-1] *= i 

        tmp = np.hstack((tmp, class_number))
        if i > 0:
           data = np.vstack((data, tmp))
        else:
            data = copy.deepcopy(tmp)

    # Normalize
    tmp = data[:,:-1] 
    # tmp = tmp - tmp.mean(axis=0)
    tmp = tmp / tmp.max(axis=0)
    data[:,:-1] = tmp

    return data

def split_data(data, training_size):
    """Splits the data into training and testing set.
        Specify the training size and the rest will be for
        testing.
    
    Arguments:
        data: np.array -- Numpy array with features in the first four columns, 
                       then a column of ones followed by a column of labels
        training_size: int -- The number of training samples for the data
    
    Returns:
        np.array -- Training data
        np.array -- Test data
    """

    N = int(data.shape[0] /CLASSES)
    test_size = N-training_size
    sample_length = data.shape[1]
    trainig_data = np.zeros((CLASSES*training_size,sample_length))
    test_data = np.zeros((CLASSES*test_size,sample_length))
    for i in range(CLASSES):
        tmp = data[(i*N):((i+1)*N)]
        trainig_data[(i*training_size):((i+1)*training_size),:] = tmp[:training_size,:]
        test_data[(i*test_size):((i+1)*test_size),:] = tmp[training_size:,:]

    return trainig_data, test_data


def plot_petal(data):
    """Plots petal length against the width for ach sample in 
        the data provided.
    
    Arguments:
        data: np.array -- Array with data loaded using load_data
    """
    for i in range(CLASSES):
        petal_data = data[(50*i):(50*(i+1)),2:-2]
        plt.scatter(petal_data[:,0],petal_data[:,1])
    plt.title("Petal data")
    plt.show()

def plot_sepal(data):
    """Plots sepal length against the width for ach sample in 
        the data provided.
    
    Arguments:
        data: np.array -- Array with data loaded using load_data
    """
    for i in range(CLASSES):
        sepal_data = data[(50*i):(50*(i+1)),:-4]
        plt.scatter(sepal_data[:,0],sepal_data[:,1])
    plt.title("Sepal data")
    plt.show()


def sigmoid(x):
    """The sigmoid function applied to all elements 
        of a np.array.
    
    Arguments:
        x: np.array -- Input array with datapoints
    
    Returns:
        np.array -- Input array with the sigmoid functio applied to it
    """
    return np.array(1 / (1 + np.exp(-x)))

def train(data, iterations, alpha, plot=False):
    """Trains a linear classifier on the provided dataset.
    
    Arguments:
        data: np.array -- training data with features and labels
        iterations: int -- The number of iterations for the training
        alpha: int -- Step size for the training
    
    Returns:
        np.array -- The classifier W with shape CxD+1
    """
    # tk are targets aka actual class, shape = (3,1)
    # g = Wx
    mse_liste = []
    features = data.shape[1]-2
    g_k = np.zeros((CLASSES))
    g_k[0] = 1
    t_k = np.zeros((CLASSES,1))
    # W = np.random.uniform(low=-10, high=10, size=(CLASSES,features+1))
    W = np.zeros((CLASSES,features+1))
    for i in range(iterations):
        grad_W_MSE = 0
        mse = 0
        # np.random.shuffle(data) # Shuffle the data before each iteration
        for x_k in data:
            # Find g_k
            tmp = np.matmul(W,(x_k[:-1]))[np.newaxis].T 
            g_k = sigmoid(tmp)


            # Extract target and update t_k
            t_k *= 0 
            t_k[int(x_k[-1]),:] = 1
            tk = t_k[np.newaxis].T


            # Eq 3.22
            grad_gk_MSE = np.multiply((g_k - t_k), g_k) 
            grad_W_zk = x_k[:-1].reshape(1,features+1)

            grad_W_MSE += np.matmul(np.multiply(grad_gk_MSE, (1-g_k)), grad_W_zk) #[np.newaxis]
            
            mse += 0.5* np.matmul((g_k-t_k).T,(g_k-t_k))
        mse_liste.append(mse[0])

        # Eq 3.23
        W = W-alpha*grad_W_MSE 

    if plot:
        plt.plot(mse_liste)
    
    return W


def get_confuse_matrix(W, test_data):
    """Computes the confusion matrix for the classifer
        and the data
    
    Arguments:
        W: np.array -- CxD+1 classifier
        test_data: np.array -- the data to be classified
                               with features and labels
    
    Returns:
        np.array -- Confusion matrix
    """
    confuse_matrix = np.zeros((CLASSES,CLASSES))
    for i in range(len(test_data)):
        prediction = int(np.argmax(np.matmul(w,test_data[i,:-1])))
        actual = int(test_data[i,-1])
        confuse_matrix[prediction,actual] += 1

    print(confuse_matrix)
    error_rate = (1 - np.sum(confuse_matrix.diagonal())/np.sum(confuse_matrix))*100
    print(f"Error rate: {error_rate}%")
    return confuse_matrix



def plot_histogram(data, step=0.03):
    """Plots the four features of the plant in one plot.
    
    Arguments:
        data: np.array -- Iris data loaded with the  
    
    Keyword Arguments:
        step: float}-- [description] (default: {0.1})
    """
    f, axis = plt.subplots(2,2, sharex='col', sharey='row')
    max_val = np.amax(data)         # Finds maxvalue in samples
    N = int(data.shape[0]/CLASSES)    # slice variables used for slicing samples by class
    
    # Create bins (sizes of histogram boxes)
    bins = np.linspace(0.0 ,int(max_val+step), num=int((max_val/step)+1), endpoint=False)

    legends = ['Class 1: Setosa', 'Class 2: Versicolour', 'Class 3: Virginica']
    colors = ['Red', 'Blue', 'lime']
    features = {0: 'sepal length',
                1: 'sepal width',
                2: 'petal length',
                3: 'petal width'}

    for feature in features:
        axis_slice = str(bin(feature)[2:].zfill(2))
        plt_axis = axis[int(axis_slice[0]),int(axis_slice[1])]
        # Slices samples by class
        samples = [data[:N, feature], data[N:2*N, feature], data[2*N:, feature]]

        # Creates plots, legends and subtitles
        for i in range(3):
            plt_axis.hist(samples[i], bins, alpha=0.5, stacked=True, label=legends[i], color=colors[i])
        plt_axis.legend(prop={'size': 7})
        plt_axis.set_title(f'feature {feature+1}: {features[feature]}')

    for ax in axis.flat:
        ax.set(xlabel='Measure [cm]', ylabel='Number of samples')
        ax.label_outer() # Used to share labels on y-axis and x-axis
    plt.show()


def plot_confusion_matrix(confusion_matrix, classes, name="Confusion matrix"):
    df_cm = pd.DataFrame(confusion_matrix, index=classes, columns=classes)
    fig = plt.figure(num=name, figsize = (5,5))
    sn.heatmap(df_cm, annot=True)
    plt.show()


if __name__ == "__main__":
    data = load_data()
    # plot_petal(data)
    # plot_sepal(data)
    # plot_histogram(data)
    # List of alphas used to tune parameters
    # alphas = [0.5, 0.25, 0.1, 0.05, 0.005]
    alphas = [0.05]
    iterations = 5000
    train_set, test_set = split_data(data,30)
    for alpha in alphas:
        print("30 train, 20 test")
        w = train(train_set[:,:],iterations, alpha, True)
        print("Training")
        conf = get_confuse_matrix(w, train_set)
        plot_confusion_matrix(conf, legends)
        print("Testing")
        conf = get_confuse_matrix(w, test_set)
        plot_confusion_matrix(conf, legends)

        print("----------")
        print("20 train, 30 test")
        w = train(test_set[:,:],iterations, alpha)
        print("Training")
        conf = get_confuse_matrix(w, test_set)
        plot_confusion_matrix(conf, legends)
        print("Testing")
        conf = get_confuse_matrix(w, train_set)
        plot_confusion_matrix(conf, legends)
   

        print("Removing features")
        print("Removing sepal width")
        train_set = np.delete(train_set,1,1) # Deleting column
        test_set = np.delete(test_set,1,1) # Deleting column
        w = train(train_set[:,:],iterations, alpha)
        print("Training")
        conf = get_confuse_matrix(w, train_set)
        plot_confusion_matrix(conf, legends)
        print("Test")
        conf = get_confuse_matrix(w, test_set)
        plot_confusion_matrix(conf, legends)

        print("Removing sepal length")
        test_set = np.delete(test_set,0,1) # Deleting column
        train_set = np.delete(train_set,0,1) # Deleting column
        w = train(train_set[:,:],iterations, alpha)
        print("Training")
        conf = get_confuse_matrix(w, train_set)
        plot_confusion_matrix(conf, legends)
        print("Test")
        conf = get_confuse_matrix(w, test_set)
        plot_confusion_matrix(conf, legends)

        print("Removing petal length")
        test_set = np.delete(test_set,0,1) # Deleting column
        train_set = np.delete(train_set,0,1) # Deleting column
        w = train(train_set[:,:],10000, 0.15) # Adjusted alpha and iterations
        print("Training")
        conf = get_confuse_matrix(w, train_set)
        plot_confusion_matrix(conf, legends)
        print("Test")
        conf = get_confuse_matrix(w, test_set)
        plot_confusion_matrix(conf, legends)

    plt.ylabel("MSE")
    plt.xlabel("Iteration number")
    alpha_list = ["Alpha: " +str(alpha) for alpha in alphas]
    plt.legend(alpha_list)

    plt.show()
