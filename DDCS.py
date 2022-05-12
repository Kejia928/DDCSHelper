import numpy as np
import matplotlib.pyplot as plt


def statistic(x):
    mean = np.mean(x)
    median = np.median(x)
    # ddof is 1, which means the sample var and std
    std = np.std(x, ddof=1)
    var = np.var(x, ddof=1)
    print('Mean: {}\nMedian: {}\nSample Standard Deviation: {}\nSample Variance: {}'.format(mean, median, std, var))
    return mean, median, std, var


def whole_static(x):
    mean = np.mean(x)
    median = np.median(x)
    std = np.std(x)
    var = np.var(x)
    print('Mean: {}\nMedian: {}\nStandard Deviation: {}\nVariance: {}'.format(mean, median, std, var))
    return mean, median, std, var


def manhattan_distance(x, y):
    # check the length
    if len(x) != len(y):
        print("The length is not same.")
        return
    else:
        distance = 0
        # loop the array
        for index in range(len(x)):
            distance = distance + np.abs(x[index] - y[index])

        # output the answer
        print('Manhattan Distance: {}'.format(distance))
        return distance


def euclidean_distance(x, y):
    # check the length
    if len(x) != len(y):
        print("The length is not same.")
        return
    else:
        distance = 0
        # loop the array
        for index in range(len(x)):
            distance = distance + np.abs(x[index] - y[index]) ** 2

        # output the answer
        distance = np.sqrt(distance)
        print('Euclidean Distance: {}'.format(distance))
        return distance


def p_norm_distance(x, y, p):
    # check the length
    if len(x) != len(y):
        print("The length is not same.")
        return
    else:
        distance = 0
        # loop the array
        for index in range(len(x)):
            distance = distance + np.abs(x[index] - y[index]) ** p

        # output the answer
        distance = np.power(distance, 1 / p)
        print('{} - norm Distance: {}'.format(p, distance))
        return distance


def chebyshev_distance(x, y):
    # check the length
    if len(x) != len(y):
        print("The length is not same.")
        return
    else:
        distance_list = []
        # loop the array
        for index in range(len(x)):
            distance_list.append(np.abs(x[index] - y[index]))

        # output the answer
        distance = np.max(distance_list)
        print('Chebyshev Distance: {}'.format(distance))
        return distance


def hamming_distance(str1, str2):
    # check the length
    if len(str1) != len(str2):
        print("The length is not same.")
        return
    else:
        distance = 0
        # loop the array
        for index in range(len(str1)):
            if str1[index] != str2[index]:
                distance = distance + 1

        # output the answer
        print('Hamming Distance: {}'.format(distance))
        return distance


def edit_distance(str1, str2):
    # if two string are empty, the edit distance is zero
    if len(str1) == 0 and len(str2) == 0:
        print('Edit Distance: {}'.format(0))
        return 0
    # if one string is empty, the edit distance equal to another string's length
    elif len(str1) == 0:
        print('Edit Distance: {}'.format(len(str2)))
        return len(str2)
    elif len(str2) == 0:
        print('Edit Distance: {}'.format(len(str1)))
        return len(str1)
    else:
        # make a matrix
        edit = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

        edit[0][0] = 0  # if two string are empty, the edit distance is zero

        # calculate distance
        for i in range(1, len(str1) + 1):
            for j in range(1, len(str2) + 1):
                if str1[i - 1] == str2[j - 1]:
                    d = 0
                else:
                    d = 1

                edit[i][j] = min(edit[i - 1][j] + 1, edit[i][j - 1] + 1, edit[i - 1][j - 1] + d)

        # get distance
        distance = edit[len(str1)][len(str2)]
        print('Edit Distance: {}'.format(distance))
        return distance


def wup_relatedness():
    print("go http://ws4jdemo.appspot.com/")
    return


def linear_regression(x, y):
    # create matrix
    # steps :
    # At the beginning, we have y with shape (len(y),)
    # use [y] to create a matrix with shape (1,len(y))
    # then, use transpose to reshape it into shape (len(y),1)
    matrix_y = np.transpose([y])
    matrix_x = np.stack((np.ones(np.shape(x)[0]), x), axis=-1)
    # use formula
    result = np.matrix(matrix_x.T @ matrix_x).I @ matrix_x.T @ matrix_y
    print('fit_Wh: ')
    print(result)
    print('Equation: y = {} + {} x'.format(result[0, 0], result[1, 0]))
    return result[0, 0], result[1, 0]


def regularised(x, y, sigma, lam=1):
    # create matrix
    matrix_x = np.empty((len(x), 2))
    matrix_y = np.empty((len(y), 1))
    # add a column of 1s in x matrix
    for i in range(0, len(x)):
        matrix_x[i][0] = 1
        matrix_x[i][1] = x[i]
        matrix_y[i][0] = y[i]

    # get reg
    reg = sigma ** 2
    # get diagonal matrix
    diagonal = np.eye(matrix_x.shape[1])
    diagonal = reg * lam * diagonal

    # use formula
    result = np.matrix(matrix_x.T @ matrix_x + diagonal).I @ matrix_x.T @ matrix_y
    print('fit_Wh: ')
    print(result)
    print('Equation: y = {} + {} x'.format(result[0, 0], result[1, 0]))
    return result[0, 0], result[1, 0]


def classification_MLE_two_class(l0, l1):
    likelihood = 0
    for i in l0:
        likelihood = likelihood + (-np.log(1 + np.e**i))
    for j in l1:
        likelihood = likelihood + (-np.log(1 + np.e**(-j)))
    print('Log-Likelihood: {}'.format(likelihood))
    return likelihood


def KNN(class0, class1, newDatapoint, K):
    distance_list = []
    nearest_point = {}

    for i in class0:
        distance = euclidean_distance(i, newDatapoint)
        if len(nearest_point) >= K:
            if np.max(distance_list) > distance:
                distance_list.remove(np.max(distance_list))
                distance_list.append(distance)
                del nearest_point[distance]
                nearest_point[distance] = 0
        else:
            distance_list.append(distance)
            nearest_point[distance] = 0

    for j in class1:
        distance = euclidean_distance(j, newDatapoint)
        if len(nearest_point) >= K:
            if np.max(distance_list) > distance:
                distance_list.remove(np.max(distance_list))
                distance_list.append(distance)
                del nearest_point[np.max(distance_list)]
                nearest_point[distance] = 1
        else:
            distance_list.append(distance)
            nearest_point[distance] = 1

    temValues = list(nearest_point.values())
    values = np.array(temValues)

    zero = 0
    one = 0
    for k in range(0, len(values)):
        if values[k] == 0:
            zero = zero + 1
        elif values[k] == 1:
            one = one + 1

    if one == zero:
        print("Predict new data point can be in class 0 and class 1")
    elif one < zero:
        print("Predict new data point in class0")
    elif zero < one:
        print("Predict new data point in class1")
    return


def weight_calculator(distance, b):
    weight = np.e**(-distance/(2*b))
    print('Weight: {}'.format(weight))
    return weight


def WNN(class0, class1, newDatapoint, b):
    weight_list = {}
    weights = []

    for i in class0:
        distance = euclidean_distance(i, newDatapoint)
        weight = weight_calculator(distance, b)
        weights.append(weight)
        weight_list[weight] = 0

    for j in class1:
        distance = euclidean_distance(j, newDatapoint)
        weight = weight_calculator(distance, b)
        weights.append(weight)
        weight_list[weight] = 1

    zero = 0
    one = 0

    totalWeight = np.sum(weights)
    for k in weights:
        if weight_list[k] == 0:
            zero = zero + (k/totalWeight)
        elif weight_list[k] == 1:
            one = one + (k/totalWeight)

    if one == zero:
        print("Predict new data point can be in class 0 and class 1")
    elif one < zero:
        print("Predict new data point in class0")
    elif zero < one:
        print("Predict new data point in class1")
    return


def logits_to_probabilities(logits):
    p = [1/1+np.exp(-logit) for logit in logits]
    print('Probability: {}'.format(p))
    return p


# KMEANS RELATED FUNCTIONS
# a__________________________ START HERE _______________________________a


def kmeans(X, K):
    N, D = X.shape
    X = X[:, None, :]
    mu = np.random.randn(K, D)
    while True:
        print("Clustering")
        sd = ((X - mu) ** 2).sum(-1)  # sd.shape = (N, K)
        z = np.argmin(sd, 1)  # z.shape  = (N)
        q = np.zeros((N, K, 1), dtype=int)
        q[np.arange(N), z, 0] = 1.
        print(f"loss = {loss(X, z, mu).item()}")
        plot(X, z, mu)
        yield None

        print("Calculating Mean")
        mu = (q * X).sum(0) / q.sum(0)
        print(f"loss = {loss(X, z, mu).item()}")
        plot(X, z, mu)
        yield None


def loss(X, z, mu):
    return ((X - mu[z, :][:, None, :]) ** 2).mean()


def plot(X, z, mu):
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.scatter(X[:, 0, 0], X[:, 0, 1], c=z)
    ax.scatter(mu[:, 0], mu[:, 1], s=100, c='r', label="cluster centers")
    ax.legend()
    print("Centers are: " + str(mu))


# a___________________________________end _______________________________a
