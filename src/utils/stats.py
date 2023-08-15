import numpy as np


def sample_from_gmm_2d(np_random_gen, weights, means, covariances, num_samples=1):
    # Choose which Gaussian to sample from
    gaussian_indices = np_random_gen.choice(len(weights), size=num_samples, p=weights)

    # Sample from the chosen Gaussians
    samples = []
    for idx in gaussian_indices:
        sample = np_random_gen.multivariate_normal(means[idx], covariances[idx])
        samples.append(sample)

    return np.array(samples)
