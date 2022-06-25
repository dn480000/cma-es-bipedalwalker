""" This module is a lite implementation of CMA ES.
The implementation follows 
Hansen, Nikolaus. "The CMA evolution strategy: A tutorial." arXiv preprint arXiv:1604.00772 (2016).
"""

import numpy as np

class CmaEs:
    """
    A lite implementation of CMA ES. The algorithm serachs for global minimum
    by repeatedly sampling from a multivariate normal distribution, and updating
    the mean and covariance

    The implemetation follows 
    Hansen, Nikolaus. "The CMA evolution strategy: A tutorial." arXiv preprint arXiv:1604.00772 (2016).

    """
    def __init__(self, mean, cov, population_size=None, mu=None):
        """Initialize `CmaEs` object

        Parameters
        ----------
            mean : numpy array
                1d array of the initial mean
            cov : numpy array
                2d array of the inital covariance matrix
            population_size : int
                population size, by default
                4 + int(3 * ln(search space dimensions))
            mu : int
                number of selected individuals in the population used to update the mean, by default
                int(population_size / 2)
        """
        self.mean = mean
        self.cov = cov
        self.n = self.mean.size
        if self.cov.shape != (self.n, self.n):
            raise ValueError('Covariance shape does not match mean')
        if population_size is not None:
            self.population_size = population_size
        else:
            self.population_size = 4 + int(3 * np.log(self.n))
        if mu is not None:
            self.mu = mu
        else:
            self.mu = self.population_size // 2 
        
        # Initialize hyperparameters with recommendated values
        self.generation = 0
        self.path_cov = np.zeros(self.n)
        self.path_step = np.zeros(self.n)
        self.step_size = 1

        self.weights = np.log(self.mu + 1) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.eff_mu = 1 / np.sum(self.weights ** 2)
        self.step_discount = (self.eff_mu + 2) / (self.n + self.eff_mu + 3)
        self.path_cov_discount = 4.0 / (self.n + 4)
        self.step_damping = 1 + \
            2 * np.maximum(0, np.sqrt((self.eff_mu - 1) / (self.n + 1)) - 1) + self.step_discount
        self.cov_discount = 2 / (self.eff_mu * (self.n + np.sqrt(2)) ** 2) + \
             (1 - 1 / self.eff_mu) * \
                np.minimum(1, (2 * self.eff_mu - 1) / ((self.n + 2) ** 2 + self.eff_mu)) 
        self.expected_path_step_length = np.sqrt(self.n) * (1 - 1 / (4 * self.n) + 1 / (21 * self.n ** 2))

        # Eigendecomposition of covariance matrix
        self.eigenval_sqrt, self.eigenvectors = np.linalg.eigh(self.cov)
        self.eigenval_sqrt = np.sqrt(self.eigenval_sqrt)
    
    def sample(self):
        """Draw samples with current mean and covariance
        Number of samples equals population size

        Parameters
        ----------
            None

        Returns
        -------
        samples : numpy array
            2d array of drawn samples 
            number of rows equals population size
            number of columns equals search space dimension
        """
        z = np.random.normal(0, 1, (self.population_size, self.n))
        return self.mean[np.newaxis,] + self.step_size * np.matmul(self.eigenval_sqrt[np.newaxis,] * z, self.eigenvectors.T)

    def update(self, samples, cost):
        """Update mean and covariance with samples and cost function evaluated on samples

        Parameters
        ----------
        samples : numpy array
            2d array of drawn samples 
            number of rows equals population size
            number of columns equals search space dimension

        cost : numpy array
            1d array of cost evaluated on population samples 
            number of elements equals population size
        """

        sorted_arg = np.argsort(cost)
        new_mean = np.sum(samples[sorted_arg[:self.mu]] * self.weights[:, np.newaxis], 0)
        mean_diff = new_mean - self.mean
        
        self.path_step = (1 - self.step_discount ) * self.path_step + \
            np.sqrt(self.step_discount * (2 - self.step_discount) * self.eff_mu) / self.step_size  * \
            np.matmul(self.eigenvectors, np.matmul(np.diag(1 / self.eigenval_sqrt), self.eigenvectors.T)) \
                .dot(mean_diff)

        h_is_one = (np.linalg.norm(self.path_step) / \
                np.sqrt(1 - (1 - self.step_discount) ** (2 * (1 + self.generation)))) < \
            ((1.5 + 1 / (self.n - 0.5)) * self.expected_path_step_length)
        self.path_cov *= 1 - self.path_cov_discount
        if h_is_one:
            self.path_cov += np.sqrt(self.step_discount * (2 - self.step_discount) * self.eff_mu) / self.step_size * mean_diff

        centered_samples = samples[sorted_arg[:self.mu]] - self.mean[np.newaxis,:]
        self.cov = (1 - self.cov_discount) * self.cov + \
            self.cov_discount / self.eff_mu * np.outer(self.path_cov, self.path_cov) + \
                self.cov_discount * (1 - 1 / self.mu) / (self.step_size ** 2) * \
                    np.matmul(centered_samples.T * self.weights, centered_samples)
                
        self.step_size *= np.exp(
            self.step_discount / self.step_damping * \
                (np.linalg.norm(self.path_step) / self.expected_path_step_length - 1))
        
        self.mean = new_mean
        self.eigenval_sqrt, self.eigenvectors = np.linalg.eigh(self.cov)
        self.eigenval_sqrt = np.sqrt(self.eigenval_sqrt)
