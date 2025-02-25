## full algorithme 


def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given dataset for T iterations.
    """
    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0.0
    update_counter = 0

    for t in range(1, T + 1):
        for i in get_order(feature_matrix.shape[0]):
            update_counter += 1
            eta = 1 / np.sqrt(update_counter)
            feature_vector = feature_matrix[i]
            label = labels[i]
            if label * (np.dot(theta, feature_vector) + theta_0) <= 1:
                theta = (1 - eta * L) * theta + eta * label * feature_vector
                theta_0 += eta * label
            else:
                theta = (1 - eta * L) * theta

    return theta, theta_0