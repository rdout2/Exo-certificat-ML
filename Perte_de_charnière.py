## hinge loss on One Datasample


def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        `feature_vector` - numpy array describing the given data point.
        `label` - float, the correct classification of the data point.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - float representing the offset parameter.

    Returns:
        the hinge loss, as a float, associated with the given data point and
        parameters.
    """
    margin = label * (np.dot(theta, feature_vector) + theta_0)
    return max(0, 1 - margin)


## The complete Hinge Loss

def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the hinge loss over a full dataset given the feature matrix, labels,
    and parameters theta and theta_0.
    """
    total_loss = 0
    num_samples = feature_matrix.shape[0]
    
    for i in range(num_samples):
        loss = hinge_loss_single(feature_matrix[i], labels[i], theta, theta_0)
        total_loss += loss

    average_loss = total_loss / num_samples
    return average_loss
