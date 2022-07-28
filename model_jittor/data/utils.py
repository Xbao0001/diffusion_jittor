def convert_to_negetive_one_positive_one(x: np.ndarray, **kwargs):
    """[0, 255] -> [-1, 1] """
    return x / 127.5 - 1.0


def to_onehot(x: np.ndarray, n_labels=29, **kwargs):
    return np.eye(n_labels)[x]