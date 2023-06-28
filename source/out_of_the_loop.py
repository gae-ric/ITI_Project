import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sampling import Metropolis

## ---------------- ##
## GAUSSIAN PROCESS ##
## ---------------- ##

class GaussianProcess():
    
    '''
    ATTRIBUTES
    ------------------------------------------
    X : [N,d] locations of the observed values
    Y : [N] observed values
    Xs : [M,d] test locations
    var : [N] fixed observation variance
    kernel : [str] kernel's name
    theta : [d+1] kernel's hyperparameters
    tuning : [dict] with key "method" for hyperparamter tuning method
    verbose : [bool]
    mu : [M] posterior mean   
    Sigma : [M,M] posterior covariance matrix  
    ------------------------------------------   
    '''
    
    def __init__(self, X, Y, Xs, var=0, kernel="matern_5", theta=np.ones(2), tuning={"method": "MLE", "bounds": None}, verbose=True):
        
   
        
        self.X = X
        self.Y = Y
        self.Xs = Xs
        self.var = var
        self.kernel = kernel
        self.theta = theta
        self.tuning = tuning
        self.verbose = verbose
        self.mu = None
        self.Sigma = None
    
    ## ----------------- ##
    ## COVARIANCE MATRIX ##
    ## ----------------- ##

    # compute matern kernel
    def matern(self, X1, X2, theta):

        dists = cdist(X1/theta[1:], X2/theta[1:], metric="euclidean")

        if "1" in self.kernel:
            K = (theta[0]**2) * np.exp(-dists)
            
        elif "3" in self.kernel:
            K = (theta[0]**2) * np.exp(-np.sqrt(3)*dists) * (1 + np.sqrt(3)*dists)

        elif "5" in self.kernel:
            K = (theta[0]**2) * np.exp(-np.sqrt(5)*dists) * (1 + np.sqrt(5)*dists + 5/3*dists**2)

        return K


    # compute RBF kernel
    def RBF(self, X1, X2, theta):
        dists = cdist(X1/theta[1:], X2/theta[1:], metric="euclidean")
        K = (theta[0]**2) * np.exp(-0.5*dists**2)

        return K


    # compute the blocks of the joint probability covariance matrix
    def cov_matrix(self, theta):

        if "matern" in self.kernel:
            K_X_X = self.matern(self.X, self.X, theta) + np.diag(self.var*np.ones(self.X.shape[0]))
            K_X_Xs = self.matern(self.X, self.Xs, theta)
            K_Xs_X = K_X_Xs.T #self.matern(self.Xs, self.X, theta)
            K_Xs_Xs = self.matern(self.Xs, self.Xs, theta)

        elif "RBF" in self.kernel:
            K_X_X = self.RBF(self.X, self.X, theta) + np.diag(self.var*np.ones(self.X.shape[0]))
            K_X_Xs = self.RBF(self.X, self.Xs, theta)
            K_Xs_X = K_X_Xs.T #self.RBF(self.Xs, self.X, theta)
            K_Xs_Xs = self.RBF(self.Xs, self.Xs, theta)

        else:
            raise ValueError("Invalid kernel name.")

        return K_X_X, K_X_Xs, K_Xs_X, K_Xs_Xs

    ## --------------------- ##
    ## HYPERPARAMETER TUNING ##
    ## --------------------- ##

    # negative log likelihood        
    def nlog_likelihood(self, theta):

        K_X_X, _, _, _ = self.cov_matrix(theta)
        
        # log likelihood
        l = - 0.5*np.log(np.linalg.det(K_X_X)) - 0.5*np.dot((self.Y).T, np.dot(np.linalg.inv(K_X_X), self.Y))
        return -l


    # compute gradient of matern kernel
    def matern_gradient(self, X1, X2, theta):

        dists = np.linalg.norm(X1/theta[1:] - X2/theta[1:])

        if theta[1:].shape[0]>1:
            B = pow(X1-X2, 2)

        else:
            B = np.sum(pow(X1-X2,2))

        if "1" in self.kernel:
            theta_0_grad = 2*theta[0] * np.exp(-dists)
            theta_grad = (theta[0]**2) * np.exp(-dists) * B/(dists*(theta[1:]**3))
            
        elif "3" in self.kernel:
            theta_0_grad = 2*theta[0] * np.exp(-np.sqrt(3)*dists) * (1 + np.sqrt(3)*dists)
            theta_grad = -theta[0]**2 * np.exp(-np.sqrt(3)*dists) * B/(dists*(theta[1:]**3))

        elif "5" in self.kernel:
            theta_0_grad = 2*theta[0] * np.exp(-np.sqrt(5)*dists) * (1 + np.sqrt(5)*dists + 5/3*(dists**2))
            theta_grad = 5/3*(theta[0]**2) * np.exp(-np.sqrt(5)*dists) * (dists - np.sqrt(5)*(dists**2)) * B/(dists*(self.theta[1:]**3))

        return np.concatenate([theta_0_grad.reshape(-1), theta_grad.reshape(-1)], axis=0)


    # compute gradient of RBF kernel
    def RBF_gradient(self, X1, X2, theta):

        dists = np.linalg.norm(X1/theta[1:] - X2/theta[1:])

        RBF_grad = np.zeros(theta.shape)
        dists_grad = - (np.sum((X1-X2)**2))/(dists*theta[1:]**3)

        RBF_grad[0] = 2*theta[0] * np.exp(-1/2*dists**2)
        RBF_grad[1:] = theta[0]**2 * np.exp(-1/2*dists**2) * dists * dists_grad

        return RBF_grad


    # compute gradient of the entire covariance matrix
    def cov_gradient(self, batch_x, theta):

        cov_grad = np.zeros((batch_x.shape[0], batch_x.shape[0], theta.shape[0]))

        for i in range(cov_grad.shape[0]):
            for j in range(cov_grad.shape[1]):

                # off-diagonal terms
                if i != j:
                    if "matern" in self.kernel:
                        cov_grad[i,j,:] = self.matern_gradient(batch_x[i], batch_x[j], theta)
                    elif "RBF" in self.kernel:
                        cov_grad[i,j,:] = self.RBF_gradient(batch_x[i], batch_x[j], theta)
                    else:
                        raise ValueError("Invalid kernel name.")
                # diagonal terms
                else:
                    diag_el = np.zeros_like(theta)
                    diag_el[0] = 2*theta[0]
                    cov_grad[i,j,:] = diag_el

        return cov_grad


    # compute gradient of the entire negative log-likelihood
    def gradient(self, batch_x, batch_y, batch_K, theta):

        K_inv = np.linalg.inv(batch_K) # [n_batch, n_batch]
        K_grad = self.cov_gradient(batch_x, theta) # [n_batch, n_batch, n_theta]

        grad = np.zeros(theta.shape[0])

        for i in range(len(grad)):
            grad[i] = 0.5*(-np.linalg.multi_dot([batch_y, K_inv, K_grad[:,:,i], K_inv, batch_y]) + np.trace(np.dot(K_inv, K_grad[:,:,i])))/batch_y.shape[0]
            
        loss = 0.5*(np.linalg.multi_dot([batch_y, K_inv, batch_y]) + np.log(np.linalg.det(batch_K)) + batch_y.shape[0]*np.log(np.pi*2))/batch_y.shape[0]

        return grad, loss


    # gradient discent algorithm
    def gradient_descent(self, theta):

        if self.verbose: print("- gradient descent started ...")

        theta_log = []
        loss_log = []

        if "batch_size" in self.tuning:
            batch_size = self.tuning.get("batch_size")
        else:
            batch_size = self.X.shape[0]

        # loop over epochs
        for j in range(self.tuning.get("epochs")):

            # loop over training points
            for i in range(0, int(self.X.shape[0]/batch_size)):

                K_X_X, _, _, _ = self.cov_matrix(theta)

                # get the sub-matrix corresponding to the batch
                batch_x = self.X[i*batch_size:(i+1)*batch_size, i*batch_size:(i+1)*batch_size]
                batch_y = self.Y[i*batch_size:(i+1)*batch_size]
                batch_K = K_X_X[i*batch_size:(i+1)*batch_size, i*batch_size:(i+1)*batch_size]

                grad, loss = self.gradient(batch_x, batch_y, batch_K, theta)
                theta = theta - (self.tuning.get("lr")/(i+1))*grad

                theta_log.append(theta)
                loss_log.append(loss)
                
        return theta_log[np.argmin(loss_log)], theta_log, loss_log


    # select tuning method and update the hyperparameters of the kernel function
    def hyperparameter_tuning(self):
        
        if self.tuning.get("method") == "MLE":

            if "bounds" in self.tuning:
                bounds = self.tuning.get("bounds")
            else:
                bounds = None

            res = minimize(self.nlog_likelihood, self.theta, method="L-BFGS-B", tol=1e-5, bounds=bounds)
            self.theta = res.x

        elif self.tuning.get("method") == "GD":

            if "epochs" in self.tuning and "lr" in self.tuning:
                theta, _, _ = self.gradient_descent(self.theta)
                self.theta = theta
            else:
                raise ValueError("Must specify epochs and learning rate.")

        elif self.tuning.get("method") == "off":
            pass

        else:
            raise ValueError("Invalid tuning method.")

    ## --------------- ##
    ## ACTIVE LEARNING ##
    ## --------------- ##

    # compute posterior mean and posterior covariance matrix
    def active_learning(self):

        self.hyperparameter_tuning()
        
        K_X_X, K_X_Xs, K_Xs_X, K_Xs_Xs = self.cov_matrix(self.theta)
        
        self.mu = np.dot(K_Xs_X, np.dot(np.linalg.inv(K_X_X), self.Y))
        self.Sigma = K_Xs_Xs - np.dot(K_Xs_X, np.dot(np.linalg.inv(K_X_X), K_X_Xs)) #+ 1e-8*np.diag(np.ones(self.Xs.shape[0]))


## --------------------- ##
## BAYESIAN OPTIMIZATION ##
## --------------------- ##

class BayesianOptimization(GaussianProcess):
    
    '''
    ATTRIBUTES
    ------------------------------------
    obj_func : [func] objective function
    acq_func : [dict]
    marginalize_acq: [dict] 
    n_query : [1] number of queries
    ------------------------------------
    '''
    
    def __init__(self, X, Y, Xs, var=0, kernel="matern_5", theta=np.ones(2), tuning={"method": "MLE", "bounds" : None}, obj_func=None, acq_func={"method": "EI", "tradeoff": 0.005}, marginalize_acq={"status": False, "num_samples": 1000, "num_burns": 300, "thinning": 5, "std": 1},  n_query=1, verbose=True):
        
        super().__init__(X=X, Y=Y, Xs=Xs, var=var, kernel=kernel, theta=theta, tuning=tuning, verbose=verbose)

        if marginalize_acq.get("status") == True and tuning.get("method") != "off":
            raise ValueError("Tuning must be off with marginalize acquisition function.")


        self.obj_func = obj_func
        self.acq_func = acq_func
        self.marginalize_acq = marginalize_acq
        self.n_query = n_query

        # history
        self.X_log = []
        self.Y_log = []
        self.Xs_log = []
        self.mu_log = []
        self.Sigma_log = []
        self.alpha_log = []
        
    ## -------------------- ##
    ## ACQUISITION FUNCTION ##
    ## -------------------- ##
    
    # probability of improvement
    def PI(self, mu, Sigma):

        if "tradeoff" in self.acq_func:
            tradeoff = self.acq_func.get("tradeoff")
        else:
            raise ValueError("Must specify tradeoff.")

        std = np.sqrt(np.diag(Sigma))
        alpha = norm.cdf((mu - np.max(self.Y) - tradeoff)/std)

        return alpha


    # expected improvement
    def EI(self, mu, Sigma):

        if "tradeoff" in self.acq_func:
            tradeoff = self.acq_func.get("tradeoff")
        else:
            raise ValueError("Must specify tradeoff.")

        std = np.sqrt(np.diag(Sigma))
        z = (mu - np.max(self.Y) - tradeoff)/std
        alpha = (mu - np.max(self.Y) - tradeoff)*norm.cdf(z) + std*norm.pdf(z)

        return alpha


    # upper condifence bound
    def UCB(self, mu, Sigma):

        if "tradeoff" in self.acq_func:
            tradeoff = self.acq_func.get("tradeoff")
        else:
            raise ValueError("Must specify tradeoff.")

        std = np.sqrt(np.diag(Sigma))
        alpha = mu + tradeoff*std

        return alpha    


    # compute acquisition function
    def acquisition_function(self, mu, Sigma):
        
        if self.acq_func.get("method") == "PI":
            alpha = self.PI(mu, Sigma)

        elif self.acq_func.get("method") == "EI":
            alpha = self.EI(mu, Sigma)

        elif self.acq_func.get("method") == "UCB":
            alpha = self.UCB(mu, Sigma)

        else:
            raise ValueError("Invalid acquisition function.")

        return alpha

    ## ----------------------------- ##
    ## MARGINAL ACQUISITION FUNCTION ##
    ## ----------------------------- ##

    # log-likelihood
    def log_likelihood(self, theta):
        return -self.nlog_likelihood(theta)


    # compute marginal acquisition
    def marginal_acquisition(self):

        samples = Metropolis(
            func=self.log_likelihood,
            initial_state=self.theta,
            num_samples=self.marginalize_acq.get("num_samples"),
            num_burns=self.marginalize_acq.get("num_burns"),
            thinning=self.marginalize_acq.get("thinning"),
            std=self.marginalize_acq.get("std")
        )

        plt.plot(samples[:,0], label="theta 0")
        plt.plot(samples[:,1], label="theta 1")
        plt.legend()
        plt.show()

        # naive update of theta
        self.theta = np.median(samples, axis=0)

        marginal_alpha = 0

        for theta in samples:

            K_X_X, K_X_Xs, K_Xs_X, K_Xs_Xs = self.cov_matrix(theta)
        
            mu = np.dot(K_Xs_X, np.dot(np.linalg.inv(K_X_X), self.Y))
            Sigma = K_Xs_Xs - np.dot(K_Xs_X, np.dot(np.linalg.inv(K_X_X), K_X_Xs))

            alpha = self.acquisition_function(mu, Sigma)
            marginal_alpha += alpha

        return marginal_alpha

    ## --------------- ##
    ## BAYESIAN SEARCH ##
    ## --------------- ##

    # Bayesian search using a Gaussian process
    def Bayesian_search(self):

        print("Bayesian search started ...")

        for i in range(self.n_query):
            if self.verbose: print(f"\nIteration {i+1}/{self.n_query}")

            # compute marginal acquisition function
            if self.marginalize_acq.get("status") == True:
                alpha = self.marginal_acquisition()

            # update posterior mean and posterior covariance matrix
            self.active_learning()
            if self.verbose: print("- theta:", self.theta)
            if self.verbose: print("- non-positive var:", np.diag(self.Sigma)[np.diag(self.Sigma) <= 0])
            
            # compute acquisition function
            if self.marginalize_acq.get("status") == False:
                alpha = self.acquisition_function(self.mu, self.Sigma)

            # compute next query point
            x_query = self.Xs[np.argmax(alpha),:]
            y_query = self.obj_func(x_query)
            if self.verbose: print(f"- query point: {x_query}")
            
            # update observed locations and values
            self.X = np.append(self.X, [x_query], axis=0)
            self.Y = np.append(self.Y, y_query)
            
            # update history
            self.X_log.append(self.X)
            self.Y_log.append(self.Y)
            self.Xs_log.append(self.Xs)
            self.mu_log.append(self.mu)
            self.Sigma_log.append(self.Sigma)
            self.alpha_log.append(alpha)

            # remove query point from test locations
            self.Xs = np.delete(self.Xs, np.argmax(alpha), axis=0)
    
## ------------------ ##
## RANDOM GRID SEARCH ##
## ------------------ ##

class RandomGridSearch(GaussianProcess):

    '''
    ATTRIBUTES
    ----------------------------------------------------
    obj_func : [func] objective function
    n_query : [1] number of queries
    verbose : [bool]
    ----------------------------------------------------
    '''

    def __init__(self, X, Y, Xs, obj_func=None, n_query=1, verbose=True):
        
        super().__init__(X=X, Y=Y, Xs=Xs, verbose=verbose)
        
        self.obj_func = obj_func
        self.n_query = n_query

        # history
        self.X_log = []
        self.Y_log = []
        self.Xs_log = []
            

    def random_search(self):

        print("Random search started ...")

        for i in range(self.n_query):
            if self.verbose: print(f"\nIteration {i+1}/{self.n_query}")

            # compute next query point
            rand_ind = np.random.randint(0, self.Xs.shape[0])
            x_query = self.Xs[rand_ind,:]
            y_query = self.obj_func(x_query)
            if self.verbose: print(f"- query point: {x_query}")
            
            # update observed locations and values
            self.X = np.append(self.X, [x_query], axis=0)
            self.Y = np.append(self.Y, y_query)
            
            # update history
            self.X_log.append(self.X)
            self.Y_log.append(self.Y)
            self.Xs_log.append(self.Xs)

            # remove query point from test locations
            self.Xs = np.delete(self.Xs, rand_ind, axis=0)