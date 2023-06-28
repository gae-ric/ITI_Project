import numpy as np


def Metropolis(func, initial_state, num_samples, num_burns, thinning, std):

    '''
    INPUT
    ---------------------------------------------------------
    func : [callable] log distribution
    initial_state : [d] initial parameter values
    num_samples : [1] number of MCMC samples to generate
    num_burns : [1]
    thinning: [1] number of samples to skip between retained samples 
    std : [1] standard deviation of the proposal distribution
    ---------------------------------------------------------
    '''

    # initialization
    current_state = initial_state.copy()

    # array to store MCMC samples
    samples = np.zeros((num_samples, current_state.shape[0]))

    # compute func on the initial state
    current_func = func(current_state)

    # generate the MCMC samples
    for i in range(num_samples):

        # generate a proposal state
        proposal_state = current_state + np.random.normal(scale=std, size=initial_state.shape)

        # compute func on the proposal state
        proposal_func = func(proposal_state)

        # calculate the acceptance probability of the proposal state
        acceptance_prob = np.exp(proposal_func - current_func)

        # accept or reject the proposal state based on the acceptance probability
        if np.random.rand() < acceptance_prob:
            current_state = proposal_state
            current_func = proposal_func

        # store the current state as a sample
        samples[i] = current_state

    # burn initial samples
    samples = samples[num_burns:,:]
    
    # sub sample to reduce autocorrelation
    samples = samples[::thinning,:]

    return samples