

def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    """
    Performs a single step update of the perceptron algorithm.
    """
    if label * (np.dot(current_theta, feature_vector) + current_theta_0) <= 0:
        current_theta += label * feature_vector
        current_theta_0 += label

    return current_theta, current_theta_0


def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm over T iterations through the given dataset.
    """
    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0.0

    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            feature_vector = feature_matrix[i]
            label = labels[i]
            theta, theta_0 = perceptron_single_step_update(feature_vector, label, theta, theta_0)

    return theta, theta_0
