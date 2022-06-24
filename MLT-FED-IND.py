import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
import random
from scipy.special import expit as sigmoid
from numpy import linalg as LA

# Main server - solving l2,1 norm
def _Solve_multitask(W0, lambda1, lambda2, max_iter, _Individual_value, rho_lst, alpha_lst, frac):
    """Objective(smooth): least squares lost(l(W)) + rho/2 * ||h(wt)||^2 + alpha * h(wt) + lambda2 * ||WR||^2 
        Input: X: {nt * d} * T 
        rho: h(wt) penalty
        lambda1: L2,1-norm group Lasso parameter
        lambda2: Mean rugularization parameter 
        _loss: function {calculate least squares loss}
        Output: W: {d * d} * T learned adjacency matrices"""

    def _Lasso_projection(W, lambda_lst, gamma):  
        Wp = np.zeros(W.shape)
        for i in range(W.shape[0]): 
            v = W[i, :]
            nm = LA.norm(v, 2) # calculate the l2 norm of v
            if nm == 0:
                w = np.zeros(v.shape)
            else:
                w = max(nm - lambda_lst[i]/gamma, 0)/nm * v
            Wp[i, :] = w
        return Wp

    ### Compute adaptive l21 norm and respective weights
    def _adp_norm_21(lambda_adp, W):
        norm_21_tot = 0
        lambda_lst = []
        # calculate weights vector
        lambda_lst = [lambda_adp * LA.norm(W[i], 2) ** (-lambda_adp) for i in range(W.shape[0])]
        norm_21_tot = sum([lambda_lst[i] * LA.norm(W[i], 2) for i in range(W.shape[0])])
        return norm_21_tot, lambda_lst 
    
    '''Algorithm procedure'''
    func_val = []  # records obj value for each iteration to check stopping condition
    bFlag = 0 # this flag tests whether the gradient step only changes a little
    d = int(W0.shape[0] ** 0.5)
    count = 0

    Wz= W0 
    Wz_old = W0

    loss_old, g_loss_old = None, None

    t = 1  
    t_old = 0 

    _ = 0 # current iter 
    gamma = 64  # line search ini value
    gamma_inc = 2 

    while _ < max_iter:
        alpha = (t_old - 1) /t  
        Ws = (1 + alpha) * Wz - alpha * Wz_old  # search point

        # Pass to individual server - STOCHASTIC
        servers_lst = random.sample(range(T), int(frac*T))
        W_loss, W_grad, loss_lst, g_loss_lst = _Individual_value(Ws, d, T, lambda2, rho_lst, alpha_lst, servers_lst, loss_old, g_loss_old)
        # function value and gradients of the search point  
        val_Ws, g_Ws  = W_loss, W_grad
        count += 1
    
        while True:
            lambda_lst = _adp_norm_21(lambda1, Ws - g_Ws / gamma)[1]
            Wzp = _Lasso_projection(Ws - g_Ws / gamma, lambda_lst, gamma) # solve EP_12 problem
            Fzp = _Individual_value(Wzp, d, T, lambda2, rho_lst, alpha_lst, range(T), loss_old, g_loss_old)[0]  # evaluate obj value for projected Wzp
            count += 1

            delta_Wzp = Wzp - Ws  # difference between Ws and Wzp(new search point) 

            r_sum = LA.norm(delta_Wzp, 'fro') ** 2  # ||delta_w||^2
            Fzp_gamma = val_Ws + sum(sum(np.multiply(delta_Wzp, g_Ws)))+ gamma/2 * r_sum

            if r_sum <= 1e-50:
                bFlag=1 # this shows that, the gradient step makes little improvement
                break
        
            if (Fzp <= Fzp_gamma) & (Fzp != np.inf): 
                break
            else:
                gamma = gamma * gamma_inc
            
        Wz_old = Wz
        Wz = Wzp   # update w
        loss_old, g_loss_old = loss_lst, g_loss_lst   # store previous function value and the gradient
        func_val.append(Fzp + _adp_norm_21(lambda1, Wz)[0])  # function value
        
        if bFlag:
            print('The program terminates as the gradient step changes the solution very small.')
            break
        elif _ >= 2 and abs(func_val[_] - func_val[_-1]) <= 1e-4:
            break
        elif _>=2 and abs(func_val[_] - func_val[_-1]) <= 1e-4 * func_val[_-1]:
            break
        elif func_val[_] <= 1e-4:
            break
            
        _ += 1
        t_old = t
        t = 0.5 * (1 + (1+ 4 * t_old ** 2) ** 0.5)  
    
    return Wzp, count
            
# Individual sub-optimization problem - return grad and individual obj value - no optimization step
def Individual_linear(X, T, lambda1, lambda2, max_iter, beta, M, h_tol=1e-8, rho_max=1e+16, w_threshold=1e-3):
    """Solve min_W L(W; X) s.t. h(wt) = 0 using augmented Lagrangian on individual servers.

    Args:
        X (np.ndarray): [nt * d] * T a list of training data
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(wt_est)| <= htol
        rho_max (float): exit if rho >= rho_max

    Returns:
        W_est (np.ndarray): [d, d] * T list of estimated DAGs
    """

    def _loss(W, i):
        """Evaluate value and gradient of Least-squares loss."""
        M = X[i] @ W
        R = X[i] - M
        loss = 0.5 / X[i].shape[0] * (R ** 2).sum()
        G_loss = - 1.0 / X[i].shape[0] * X[i].T @ R
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint of [d * d] matrix."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h
    
    # Version 1 --- INDIVIDUAL setting
    def _func(W, i, rho, alpha):
        """Evaluate individual server's loss, gradient."""
        #i, rho, alpha = args[0], args[1], args[2]
        #W = W.reshape([d, d])
        loss, G_loss = _loss(W, i)  # Calculate loss and the gradient of least squares loss
        h, G_h = _h(W) # Acyclicity constraint for W
        obj = loss + 0.5 * rho * h * h + alpha * h 
        g_obj = G_loss + (rho * h + alpha) * G_h 
        return obj, g_obj

    def _Individual_value(W, d, T, lambda2, rho_lst, alpha_lst, servers_lst, loss_old, g_loss_old):
        ''' Get individual loss, grad'''
        if loss_old == None:
            g_loss_lst, loss_lst = [], []
            for i in range(T):
                loss, G_loss = _func(W[:, i].reshape([d, d]), i, rho_lst[i], alpha_lst[i])
                g_loss_lst.append(G_loss)
                loss_lst.append(loss)
        else:  # Update the selected servers
            loss_lst, g_loss_lst = loss_old, g_loss_old
            for i in servers_lst:
                loss, G_loss = _func(W[:, i].reshape([d, d]), i, rho_lst[i], alpha_lst[i])
                g_loss_lst[i] = G_loss
                loss_lst[i] = loss
        W_loss, W_grad = _aggregate_server(d, T, g_loss_lst, loss_lst, lambda2, W)
        return W_loss, W_grad, loss_lst, g_loss_lst
    
    def _aggregate_server(d, T, g_loss_lst, loss_lst, lambda2, W):
        '''Aggregate and calculate the search point's gradient and loss for each individual task'''
        W_grad, W_loss = np.zeros([d * d, T]), 0
        R = np.identity(T) - np.ones((T, T))/T
        # Combine individual gradient, obj value and weight matrices
        for i in range(T):
            W_grad[:, i] = g_loss_lst[i].reshape([1, d * d])
            W_loss += loss_lst[i]
        # Add the mean regularization term
        W_loss += lambda2 * LA.norm(W @ R, 'fro') ** 2 
        W_grad += 2 * lambda2 * W @ R @ R.T 
        return W_loss, W_grad

    def _aggregate_w(d, T, lst):
        W = np.zeros([d * d, T])
        for i in range(T):
            W[:, i] = lst[i].reshape([1, d * d])
        return W
    
    '''@Paramaters of individual server'''
    d = X[0].shape[1]  # sample size & dimension of the current dataset
    rho, alpha, h = 1.0, 0.0, np.inf  
    X = [X[i] - np.mean(X[i], axis=0, keepdims=True) for i in range(T)]

    '''@Paramaters of all the individual servers'''
    rho_lst, alpha_lst, h_lst, w_est_lst, dual_sign_lst = [rho] * T, [alpha] * T, [h] * T, [np.zeros([d, d])] * T, [0] * T

    _ = 0 # iter number
    count_tot = []
    while (set(dual_sign_lst) != {1}) & (_ <= max_iter):   
        _ += 1 
        W_new, h_new_lst = np.zeros([d * d, T]), [None] * T
        pri_sign_lst = [0] * T
        ini_round = 1
        
        while set(pri_sign_lst) != {1}:    # Not all the individual servers have found solution for primal problem 
            if M !=0:  # Do grad desct before optimization at main server
                if ini_round != 1:  # Not initial round
                    for i in range(T):
                        if pri_sign_lst[i] != 1:
                            w_curr = W_new[:, i].reshape([d,d])
                            for j in range(M):  # M iterations of grad descent
                                grad = _func(w_curr, i, rho_lst[i], alpha_lst[i])[1]
                                w_new = w_curr - beta * grad  # Do individual gradient descent (if beta=0, no change compared to original method)
                            num = max(max(w_new.reshape([1,d * d])))
                            if num < 0.5:  # no overfitting risk
                                w_est_lst[i] = w_new
                            else:
                                w_est_lst[i] = w_curr

            W = _aggregate_w(d, T, w_est_lst)   # Combine the weight matrices list to an aggregated matrix W
            W_new, count = _Solve_multitask(W, lambda1, lambda2, 1000, _Individual_value, rho_lst, alpha_lst,frac=0.5) # Pass to the main server to optimize
            ini_round = 0
            count_tot.append(count)
            h_new_lst = [_h(W_new[:, i].reshape([d, d]))[0] for i in range(T)]

            for i in range(T):
                if (h_new_lst[i] > 0.25 * h_lst[i]) & (rho_lst[i] < rho_max):   
                    rho_lst[i] *= 10
                else:
                    w_est_lst[i], h_lst[i] = W_new[:, i].reshape([d, d]), h_new_lst[i]
                    alpha_lst[i] += rho_lst[i] * h_lst[i]  # Dual ascent
                    pri_sign_lst[i] = 1     # Individual i meets the stopping criteria for primal problem
                    if h_lst[i] <= h_tol or rho_lst[i] >= rho_max:   # Existing condition
                        dual_sign_lst[i] = 1 

    # Thresholding for extreme small coefficients
    for w in w_est_lst:
        w[np.abs(w) < w_threshold] = 0   
    total = sum(count_tot)
    return w_est_lst


# Simulation
import utils
utils.set_random_seed(2)
n, d, s0, graph_type, sem_type = 1000, 20, 20, 'ER', 'gauss'   # Gaussian noise
B_true = utils.simulate_dag(d, s0, graph_type) # True binary graph
W_true = utils.simulate_parameter(B_true) # True weighted graph
#np.savetxt('W_true_10.csv', W_true, delimiter=',')

X = [] # data list
T = 10 # number of data
for i in range(T):
    X.append(utils.simulate_linear_sem(W_true, n, sem_type)) 

W_est = Individual_linear(X, T, 0.25, 0.1, max_iter=100, beta=1e-2, M=2)
W_est 

acc = []
for i in range(T): 
    assert utils.is_dag(W_est[i])
    acc.append(utils.count_accuracy(B_true, W_est[i] != 0))
acc

for i in range(T):
    W = W_est[i]
    np.savetxt('W_est_{0}.csv'.format(i), W, delimiter=',')