from builtins import range
from builtins import object
import numpy as np
from cs231n.classifiers import KNearestNeighbor
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

                dists[i][j] = np.sqrt(np.sum(np.square(num_test[i]-self.X_train[j])))

                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            dists[i] = np.sqrt(np.sum(np.square(X[i]-self.X_train, axis=1)))
            # Numpy에서 Broadcasting은 크기가 서로 다른 배열 간의 연산을 자동으로 확장하여 계산하는 기능
            # X[i].shape = (1,5000)  self.X_train = (5000, 3072) 이기에 Broadcasting을 통해 연산을 가능하게 한다.
            # axis = 1을 지정 하면 각 행에 대한 거리를 합산하여 5000개의 거리를 표현하게 된다.

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dists = np.sqrt(-2*np.dot(X, self.X_train.T) + np.sum(np.square(self.X_train), axis=1)
                        // + np.transpose([np.sum(np.square(X), axis=1)]))

        # (A-B)(A-B) = A^2 + B^2 - 2AB를 이용해서 no loop에서도 distance를 구할 수 있다.
        #  A와 B를 각각 training data, test data의 행렬로 생각하고 전체적인 연산을 구하는 과정이다.
        # dist = np.sqrt(A^2-B^2)이기에 전체 행렬의 대한 연산을 통해서 dists를 구할 수 있다.
        # HINT에서 나온 거 처럼 한번의 multiplacation, 두번의 broadcast sums이 된다.

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def cross_validation(selfs):
        num_folds = 5
        k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

        X_train_folds = []
        y_train_folds = []
        ################################################################################
        # TODO:                                                                        #
        # Split up the training data into folds. After splitting, X_train_folds and    #
        # y_train_folds should each be lists of length num_folds, where                #
        # y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
        # Hint: Look up the numpy array_split function.                                #
        ################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        X_train_folds = np.split(selfs.X_train,num_folds)
        Y_train_folds = np.split(selfs.y_train,num_folds)
        #각각 num_folds만큼의 fold수로 나뉘어지게 된다.


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # A dictionary holding the accuracies for different values of k that we find
        # when running cross-validation. After running cross-validation,
        # k_to_accuracies[k] should be a list of length num_folds giving the different
        # accuracy values that we found when using that value of k.
        k_to_accuracies = {}

        ################################################################################
        # TODO:                                                                        #
        # Perform k-fold cross validation to find the best value of k. For each        #
        # possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
        # where in each case you use all but one of the folds as training data and the #
        # last fold as a validation set. Store the accuracies for all fold and all     #
        # values of k in the k_to_accuracies dictionary.                               #
        ################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        for k in k_choices:
            acc_list = []
            for fold in range(num_folds):
                x_val = X_train_folds[fold]  # 현재 fold를 검증 데이터로 사용
                x_t = X_train_folds[:fold] + X_train_folds[fold+1:]
                y_val = y_train_folds[fold]
                y_t = y_train_folds[:fold] + y_train_folds[fold+1:]

                x_t = x_t.reshape(-1, selfs.X_train.shape[-1])  # 2D 배열로 변환
                y_t = y_t.reshape(-1)

                classifier.train(x_t, y_t)  # k-NN 학습
                hypothesis = classifier.predict(x_val, k)  # 예측 수행

                acc = np.mean(hypothesis == y_val)  # 정확도 계산
                # boolean array 형태로 return 된다.
                # 1 or 0의 array형태로 mean을 거치게 되면서 accuracy를 계산할 수 있다.

                acc_list.append(acc)  # 정확도 저장
            k_to_accuracies[k] = acc_list  # 각 k 값에 대한 정확도 리스트 저장

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Print out the computed accuracies
        for k in sorted(k_to_accuracies):
            for accuracy in k_to_accuracies[k]:
                print('k = %d, accuracy = %f' % (k, accuracy))

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance between the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            closest_y = self.y_train[np.argsort(dists[i])[:k]]
            # np.argsort는 오름차순으로 행렬의 index number를 return하는 역할이다.
            # self.y_train을 직접 끌고와서 거리가 가장 가까운 index number를 받아 해당하는 class를 store한다.

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            y_pred[i] = np.argmax(np.bincount(closest_y))
            # np.bincount() method는 list내에 각 값들의 count를 반환한다.
            # argmax를 통해 가장 count수가 많은 class를 y_pred[i]에 저장한다.

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
