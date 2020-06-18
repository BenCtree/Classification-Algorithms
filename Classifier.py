import numpy as np
import sys
import csv
import random

def knn(training_data, test_data, k):
    # For each new example in test data
    predictions = []
    for instance_test in test_data:
        # Use get_distances to return list of tuples
        # of form (distance, training row class) sorted by distance
        sorted_dist = get_distances(training_data, instance_test)
        # Select first k elements
        first_k = sorted_dist[0:k]
        # Get class of each of the k nearest training rows
        classes = []
        for elem in first_k:
            classes.append(elem[1])
        # Make Prediction
        prediction = class_vote(classes)
        predictions.append(prediction)
    #print(predictions)
    return predictions

# Helper Functions

def euclidean_dist(vect1, vect2):
    if len(vect1) - 1 != len(vect2):
        # Vectors must be same length.
        return
    else:
        res = 0
        for i in range(len(vect2)):
            res += (vect1[i] - vect2[i])**2
        return np.sqrt(res)

def get_distances(training_data, instance_test):
    distances = []
    train_index = 0
    # For each example of training data
    for instance_train in training_data:
        # Calculate distance between new example and training example
        dist = euclidean_dist(instance_train, instance_test)
        train_class = instance_train[-1]
        # Add tuples containing distance and class of training row to distances list
        distances.append((dist, train_class))
        train_index += 1
    # Sort distances list by dist - sorted will sort by first elem of each tuple
    return sorted(distances)

def class_vote(classes):
    prediction = ''
    # See which class is more common
    yes_count = 0
    no_count = 0
    for elem in classes:
        if elem == 'yes':
            yes_count += 1
        if elem == 'no':
            no_count += 1
    if yes_count > no_count:
        #print('yes')
        prediction += 'yes'
        print(prediction)
    elif no_count > yes_count:
        #print('no')
        prediction += 'no'
        print(prediction)
    else: # yes_count == no_count
        #print('yes')
        prediction += 'yes'
        print(prediction)
    return prediction

# Notation: We let y represent the class of an observation (last elem of row of data)
# and xi represent the ith predictor variable of that observation (ith element of row)
def naive_bayes(training_data, test_data):
    predictions = []
    for instance_test in test_data:
        # Number of predictor columns in training data
        training_cols = len(training_data[0]) - 1

        # Estimate the probability of observing each class from training data
        proportion_yes, proportion_no = class_probabilities(training_data)

        # Get means and SDs for each column for each class
        mean_yes_list = []
        for i in range(training_cols):
            mean = mean_y(training_data, i, 'yes')
            mean_yes_list.append(mean)

        mean_no_list = []
        for i in range(training_cols):
            mean = mean_y(training_data, i, 'no')
            mean_no_list.append(mean)

        sd_yes_list = []
        for i in range(training_cols):
            sd = sd_y(training_data, i, 'yes', mean_yes_list[i])
            sd_yes_list.append(sd)

        sd_no_list = []
        for i in range(training_cols):
            sd = sd_y(training_data, i, 'no', mean_no_list[i])
            sd_no_list.append(sd)

        # def calc_prob(instance_test, mean_list, sd_list):
        prob_yes = calc_prob(proportion_yes, instance_test, mean_yes_list, sd_yes_list)
        prob_no = calc_prob(proportion_no, instance_test, mean_no_list, sd_no_list)

        # Make prediction
        prediction = ''
        if prob_yes > prob_no:
            prediction += 'yes'
            print(prediction)
            #print('yes')
        elif prob_no > prob_yes:
            prediction += 'no'
            print(prediction)
            #print('no')
        else: # prob_yes == prob_no
            prediction += 'yes'
            print(prediction)
            #print('yes')
        predictions.append(prediction)

    return predictions

# Helper Functions

# Notation:
# xi is the value of the ith element in the test observation
# mu_y is the mean of the ith predictor variable for class y
# sig_iy is the standard deviation of the ith predictor variable for class y
# As we are working with numerical data,
# we use the PDF of the normal distribution to calculate P(xi|y)
def normal_dist(xi, mu_y, sig_y):
    sig_sqrt2pi = np.sqrt(2*np.pi)*sig_y
    return np.exp(-(xi - mu_y)**2/(2*sig_y**2))/sig_sqrt2pi

# Given a class y (with values y = yes or y = no)
# Calculate the mean value for a particular column in the dataset
# Using just the rows with that class value
def mean_y(training_data, col_index, class_y):
    sum_over_col = 0
    count = 0
    for row in training_data:
        if row[-1] == class_y:
            sum_over_col += row[col_index]
            count += 1
    return sum_over_col / count

# Given a class y (with values y = yes or y = no)
# Calculate the standard deviation for a particular column in the dataset
# Using just the rows with that class value
def sd_y(training_data, col_index, class_y, mean):
    sum_sq_diffs = 0
    count = 0
    for row in training_data:
        if row[-1] == class_y:
            sum_sq_diffs += (row[col_index] - mean)**2
            count += 1
    # Avoid divide by 0 error when dividing sum_sq_diffs by n-1
    #if count > 1:
    #    count = count - 1
    return np.sqrt(sum_sq_diffs / (count - 1))

def class_probabilities(training_data):
    count_yes = 0
    count_no = 0
    total_obs = len(training_data)
    for row in training_data:
        if row[-1] == 'yes':
            count_yes += 1
        elif row[-1] == 'no':
            count_no += 1
    prob_yes = count_yes / total_obs
    prob_no = count_no / total_obs
    return prob_yes, prob_no

# prob_class is P(yes) or P(no) as in the proportion of the total training examples
def calc_prob(prob_class, instance_test, mean_list, sd_list):
    total_prob = 1
    for i in range(len(instance_test)):
        # Convert product of probabilities into sum of log probabilities
        # Then return the exponential of this sum to get the total probability
        # Handles underflow from product of many small numbers
        prob_xi = np.log(normal_dist(instance_test[i], mean_list[i], sd_list[i]))
        total_prob += prob_xi
    return np.exp(total_prob) * prob_class

# Stratified 10-Fold Cross Validation

def strat_10_fold_split(training_data):
    # Split training data by class
    yes_rows = []
    no_rows = []
    for row in training_data:
        if row[-1] == 'yes':
            yes_rows.append(row)
        elif row[-1] == 'no':
            no_rows.append(row)
    # Randomise rows in each class list
    random.seed(0)
    rand_yes = random.sample(yes_rows, len(yes_rows))
    rand_no = random.sample(no_rows, len(no_rows))
    # Split each into 10 folds
    yes_folds = split_list(rand_yes, 10)
    no_folds = split_list(rand_no, 10)
    # Combine pairs of yes and no folds
    folds = []
    for i in range(10):
        fold = yes_folds[i] + no_folds[i]
        # Randomise order of rows in fold
        rand_fold = random.sample(fold, len(fold))
        folds.append(rand_fold)
    return folds

def split_list(my_list, num_folds):
    remaining = len(my_list)
    fold_size = len(my_list) // num_folds
    folds = []
    last_index = 0
    for i in range(0, len(my_list), fold_size):
        if len(folds) < num_folds:
            fold = my_list[i : i+fold_size]
            folds.append(fold)
            remaining -= fold_size
            last_index = i + fold_size
    for fold in folds:
        if last_index < len(my_list):
            fold.append(my_list[last_index])
            last_index += 1
        if remaining != 0:
            remaining -= 1
        else:
            break
    return folds

def write_to_csv(folds):
    # Convert each fold to a string and append to fold_list
    fold_list = []
    for i in range(len(folds)):
        fold_string = f'fold{i+1}\n'
        for row in folds[i]:
            fold_string += list_to_string(row).replace('\'','').strip(',') + '\n'
        fold_string += '\n'
        fold_list.append(fold_string)
    # Write to csv file
    with open('pima-folds.csv','w') as csvfile:
        for fold in fold_list:
            csvfile.write(fold)

def list_to_string(my_list):
    string = ''
    for elem in my_list:
        string += str(elem) + ','
    return string

def strat_cross_val(folds, algorithm, k=0):
    num_folds = len(folds)
    accuracy_sum = 0
    # For each fold, use it as test set and each other fold as training set
    for i in range(num_folds):
        training_folds = []
        test_fold = []
        test_labelled = folds[i]
        # Remove labels from test set
        for obs in test_labelled:
            obs_nolab = obs[:-1]
            test_fold.append(obs_nolab)
        # Create training set
        for j in range(num_folds):
            if i != j:
                training_folds += folds[j]
        predictions = []
        if algorithm == 'NB':
            predictions += naive_bayes(training_folds, test_fold)
        elif algorithm == '{}NN'.format(k):
            predictions += knn(training_folds, test_fold, k)
        correct = 0
        for m in range(len(predictions)):
            if predictions[m] == test_labelled[m][-1]:
                correct += 1
        accuracy = (correct / len(test_labelled)) * 100
        accuracy_sum += accuracy
    return accuracy_sum / num_folds

def main(training_data, test_data, algorithm):

    training = []
    test = []

    # Create list of lists containing rows of training data
    with open(training_data) as training_csv:
        training_reader = csv.reader(training_csv, delimiter=' ')
        for row in training_reader:
            row_list = row[0].split(',')
            # Convert all but last element of each list to a float
            float_list = [float(elem) for elem in row_list[0:len(row_list)-1]]
            float_list.append(row_list[-1])
            training.append(float_list)

    # Create list of lists containing rows of test data
    with open(test_data) as test_csv:
        test_reader = csv.reader(test_csv, delimiter=' ')
        for row in test_reader:
            row_list = row[0].split(',')
            # Convert each element of each list to a float
            float_list = [float(i) for i in row_list]
            test.append(float_list)

    # Get k for use in KNN method call
    k = 0
    char_list = list(algorithm)
    if char_list[-1] == 'N':
        digits = char_list[0:-2]
        k = int(''.join(digits))

    # Split training data into 10 stratified folds
    folds = strat_10_fold_split(training)

    # Pick which algorithm to use
    if algorithm == 'NB':
        naive_bayes(training, test)
    elif algorithm == '{}NN'.format(k):
        knn(training, test, k)

if __name__ == "__main__":
   main(training_data=sys.argv[1], test_data=sys.argv[2], algorithm=sys.argv[3])
