##Perceptron Single Step Update
def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Computes the hinge loss on a single data point given the feature vector, label,
    and parameters theta and theta_0.
    """
    loss = max(0, 1 - label * (np.dot(theta, feature_vector) + theta_0))
    return loss


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Computes the hinge loss on a dataset.
    """
    total_loss = 0
    for i in range(len(labels)):
        total_loss += hinge_loss_single(feature_matrix[i], labels[i], theta, theta_0)

    average_loss = total_loss / len(labels)
    return average_loss


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
