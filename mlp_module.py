import numpy as np                   # for the operations with matrices and vectors
import matplotlib.pyplot as plt      # for visualizations
import matplotlib
matplotlib.use('TkAgg')  # specify a backend (here using TkAgg)
from scipy.stats import levy_stable
# the functions beta and svalue are used for generating samples from the stable distribution
def beta(a, b):
    return np.arctan(b*np.tan(np.pi*a/2))/a

def svalue(alpha, bett):
    val_beta = bett*bett
    val_tan  = np.tan(np.pi*alpha/2)*np.tan(np.pi*alpha/2)
    power_val = 1/(2*alpha)
    return (1+val_beta*val_tan)**power_val

# Generation of samples from S_{alpha}(1,0,0) denotes as SaS
def stable_sample_sym(alpha, N):
    bt = 0
    sample_exp = np.random.exponential(scale=1, size=N)
    sample_uni = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=N)
    Y1 = svalue(alpha, bt)*np.sin(alpha*(sample_uni + beta(alpha, bt)))
    Y2 = np.cos(sample_uni)
    Y2 = Y2**(-1/alpha)
    Y3 = np.cos(sample_uni - alpha*(sample_uni + beta(alpha, bt)))
    Y3 = Y3**((1-alpha)/alpha)
    Y4 = sample_exp**((alpha-1)/alpha)
    Y = Y1*Y2
    Y *= Y3
    Y *= Y4
    return Y

# Generation of samples from S_{alpha}(2^(1/alpha),0,0) denoted as SaS_2
"""If X1 and X2 are iid from S_{alpha}(1,betam mu)
   then X1 - X2 has distribution SaS_2 """
def stable_sample_symc_xmx(alpha, N):
    bt = 0
    sample_exp = np.random.exponential(scale=1, size=N)
    sample_uni = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=N)
    Y1 = svalue(alpha, bt)*np.sin(alpha*(sample_uni + beta(alpha, bt)))
    Y2 = np.cos(sample_uni)
    Y2 = Y2**(-1/alpha)
    Y3 = np.cos(sample_uni - alpha*(sample_uni + beta(alpha, bt)))
    Y3 = Y3**((1-alpha)/alpha)
    Y4 = sample_exp**((alpha-1)/alpha)
    Y = Y1*Y2
    Y *= Y3
    Y *= Y4
    return 2**(1/alpha)*Y

# Generation of samples from S_{alpha}(1,beta,0)
def stable_sample(alpha, bt, N):
    sample_exp = np.random.exponential(scale=1, size=N)
    sample_uni = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=N)
    Y1 = svalue(alpha, bt)*np.sin(alpha*(sample_uni + beta(alpha, bt)))
    Y2 = np.cos(sample_uni)
    Y2 = Y2**(-1/alpha)
    Y3 = np.cos(sample_uni - alpha*(sample_uni + beta(alpha, bt)))
    Y3 = Y3**((1-alpha)/alpha)
    Y4 = sample_exp**((alpha-1)/alpha)
    Y = Y1*Y2
    Y *= Y3
    Y *= Y4
    return Y



def empirical_distribution_function(x, point):
    """
    Calculate the empirical distribution function from a sample x at a given point.

    Parameters:
    x (numpy.ndarray): The sample for which to calculate the EDF.
    point (float or int): The point at which to evaluate the EDF.

    Returns:
    float: The EDF value at the specified point.
    """

    # Sort the sample in ascending order.
    sorted_x = np.sort(x)

    # Calculate the proportion of values less than or equal to the point of interest.
    ecdf = np.sum(sorted_x <= point) / len(sorted_x)

    return ecdf


def symmetry_deviation(N, alpha, N_samples):
    F_X0 = []
    for _ in range(N_samples):
        X = stable_sample(alpha, 0, N)
        y = empirical_distribution_function(X, 0)
        F_X0.append(y)
    return [np.quantile(F_X0, 0.05), np.quantile(F_X0, 0.95)]

def symmetry_identifier(F0, y_min_max):
    if F0 > min(y_min_max) and F0 < max(y_min_max):
        symmetric = 1
        print('The sample X is symmetric')
    else:
        symmetric = 0
        print('The sample X is not symmetric')
    return symmetric



# MLP for SaS unoptimized

def A(a, T):
    n = T.shape[0]
    A, B = np.meshgrid(T, T)
    matrix = np.exp(-abs(A + B)**a) + np.exp(-abs(A - B)**a)
    return matrix

def B(a, T):
    T = np.array(T, dtype=np.float64)
    T[T == 0] = np.finfo(np.float64).eps
    y = -abs(T)**a * np.log(abs(T)) * np.exp(-abs(T)**a)
    return y

def F(X, T):
    X1 = np.outer(X, T)
    Y1 = np.cos(X1)
    FX = np.mean(Y1, axis=0)
    return 2*FX

def projection(FX, T, a):
    return abs(np.dot(np.dot(np.linalg.inv(A(a, T)), B(a, T)), FX))

def binary_search_min_of_function_aux(fun, start, end):
    left, right = start, end
    epsilon = 1e-10  # tolerance for the precision of the minimum
    while (right - left) > epsilon:
        mid = (left + right) / 2
        if fun(mid) <= fun(mid + epsilon):
            right = mid
        else:
            left = mid
    return left, fun(mid)

def extended_binary_search(fun, start, end):
    LIST = []
    arr = np.linspace(start, end, 10)
    for i in range(len(arr)):
        if i < len(arr) - 1:
            alph, J = binary_search_min_of_function_aux(fun, arr[i], arr[i+1])
            LIST.append([alph, J])
        else:
            if arr[i] < 2:
                alph, J = binary_search_min_of_function_aux(fun, arr[i], 2)
                LIST.append([alph, J])
    min_second = min(item[1] for item in LIST)
    INDEX = [item[1] for item in LIST].index(min_second)
    output = LIST[INDEX][0]
    return output

def straightforward_search(fun, start, end):
    Alpha = list(np.arange(start_alpha, end_alpha, 0.01))+[]
    crit = []
    for al in Alpha:
        y = fun(al)
        crit.append(y)
    argmin = np.argmin(crit)
    return Alpha[argmin]

def truncate_string(num):
    num = round(num, 4)
    num = str(num)
    new_string = num[:6]
    return new_string + "0" * (6 - len(new_string))


portfolio_of_functions = {}
portfolio_of_functions.update({'sin(x)       ': np.vectorize(lambda x: np.sin(x))})
portfolio_of_functions.update({'x            ': np.vectorize(lambda x: x)})
portfolio_of_functions.update({'x**2         ': np.vectorize(lambda x: x*x)})
portfolio_of_functions.update({'x**3         ': np.vectorize(lambda x: x*x*x)})
portfolio_of_functions.update({'x**4         ': np.vectorize(lambda x: x*x*x*x)})
portfolio_of_functions.update({'x**5         ': np.vectorize(lambda x: x**5)})
portfolio_of_functions.update({'min(x**2,x)  ': np.vectorize(lambda x: min(x*x,x))})
portfolio_of_functions.update({'min(x**3,x)  ': np.vectorize(lambda x: min(x*x*x,x))})
portfolio_of_functions.update({'min(x^3,x^5) ': np.vectorize(lambda x: min(x**3,x**5))})
portfolio_of_functions.update({'sqrt(x)      ': np.vectorize(lambda x: min(x*x,x))})
portfolio_of_functions.update({'cos(x)       ': np.vectorize(lambda x: np.cos(x)*(x != 0)  )})
portfolio_of_functions.update({'x*exp(-x)    ': np.vectorize(lambda x: (x>1)*x**2+x*np.exp(-x))})
portfolio_of_functions.update({'x**4*exp(-x) ': np.vectorize(lambda x: (x>1)*x**4+x*np.exp(-x))})
KEY = list(portfolio_of_functions.keys())
fun = KEY[8]

def grid_search_choice(n):
    if n<0:
        n = 0
    elif n > len(portfolio_of_functions):
        n = len(portfolio_of_functions) - 1
    else:
        n = int(n)
    fun = KEY[n]
    print(fun)
    return portfolio_of_functions[fun]

def mlp_ebs_sas(X, T):
    start_alpha = 0.15
    end_alpha = 2
    FX = F(X, T)
    fun0 = lambda x: projection(FX, T, x)
    alpha_est = extended_binary_search(fun0, start_alpha, end_alpha)
    return alpha_est










# Computation for S_{\alpha}(1, \beta, 0)


def psi(t, a, b, p):
    return np.exp(-np.abs(t)**a*(1 - np.sign(t)*np.tan(a*p/2)*b*1j))

def dpsi_da(t, a, b, p):
    return np.gradient(psi(t, a, b, p), a)


def dpsi_db(t, a, b, p):
    return psi(t, a, b, p)*1j*np.tan(a*p/2)*np.abs(t)**a*np.sign(t)

def A_ab(a, b, T):
    T1, T2 = np.meshgrid(T, T)
    matrix = psi(T1 + T2, a, b, np.pi)
    #matrix = np.exp(-abs(T1 + T2)**a) + np.exp(-abs(T1 - T2)**a)
    return matrix


def B_a(a, b, T):
    T = np.array(T, dtype=np.float64)
    T[T == 0] = np.finfo(np.float64).eps
    return dpsi_da(T, a, b, np.pi)

def B_b(a, b, T):
    T = np.array(T, dtype=np.float64)
    T[T == 0] = np.finfo(np.float64).eps
    return dpsi_db(T, a, b, np.pi)

def F_ab(X, T):
    X1 = np.outer(X, T)
    Y1 = np.cos(X1) + 1j*np.sin(X1)
    FX = np.mean(Y1, axis=0)
    return FX

def projection_a(FX, T, a, b):
    y = np.dot(np.dot(np.linalg.inv(A_ab(a, b, T)), B_a(a, b, T)), FX)
    y = abs(np.real(y)) + abs(np.imag(y))   #y = np.real(y)
    return y

def projection_b(FX, T, a, b):
    y = np.dot(np.dot(np.linalg.inv(A_ab(a, b, T)), B_b(a, b, T)), FX)
    y = abs(np.real(y)) + abs(np.imag(y))   #y = np.real(y)
    return y

def projection_ab(FX, T, a, b):
    y1 = np.dot(np.dot(np.linalg.inv(A_ab(a, b, T)), B_a(a, b, T)), FX)
    y2 = np.dot(np.dot(np.linalg.inv(A_ab(a, b, T)), B_b(a, b, T)), FX)
    y = np.abs(y1) + np.abs(y2)
    #y = abs(np.real(y)) + abs(np.imag(y))
    return y

def projection_ab_advanced(FX, FX1, T, T1, a, b):
    projection1 = np.dot(np.linalg.inv(A_ab(a, b, T)), B_a(a, b, T))
    projection2 = np.dot(np.linalg.inv(A_ab(a, b, T1)), B_b(a, b, T1))
    max1 = max(projection1)
    max2 = max(projection2)
    try:
        if abs(max1 - 0.001) > 0.01:
            projection1 /= max1
        projection2 /= max2
        y1 = np.dot(projection1, FX)
        y2 = np.dot(projection2, FX1)
        y = np.abs(y1) + np.abs(y2)
        return y
    except RuntimeWarning:
        return 10**10


# def projection_ab_advanced2(FX1, FX2, T, T1, a, b):
#     projection1 = np.dot(np.linalg.inv(A_ab(a, b, T)), B_a(a, b, T))
#     projection2 = np.dot(np.linalg.inv(A_ab(a, b, T1(a))), B_b(a, b, T1(a)))
#     projection1 /= max(projection1)
#     projection2 /= max(projection2)
#     y1 = np.dot(projection1, FX1)
#     y2 = np.dot(projection2, FX2)
#     y = np.abs(y1) + np.abs(y2)
#     return y

def fisher_info_a(T, a, b):
    y = np.abs(np.dot(np.dot(np.linalg.inv(A_ab(a, b, T)), B_a(a, b, T)), B_a(a, b, T)))
    return abs(y)

def fisher_info_b(T, a, b):
    y = np.abs(np.dot(np.dot(np.linalg.inv(A_ab(a, b, T)), B_b(a, b, T)), B_b(a, b, T)))
    return abs(y)







def beta_estimate_functions(T, alpha, X):
    Beta = list(np.arange(-1, 1, 0.01))
    FX = F_ab(X, T)
    crit = []
    for bt in Beta:
        y = projection_b(FX, T, alpha, bt)
        crit.append(y)
    argmin = np.argmin(crit)
    beta_straighforward = Beta[argmin]
    return beta_straighforward

def beta_estimate_error_functions(T, alpha, test_beta, N, verbose):
    X = stable_sample(alpha, test_beta, N)
    Beta = list(np.arange(-1, 1, 0.01))+[]
    FX = F_ab(X, T)
    FX0 = []
    for t in T:
        FX0.append(psi(t, alpha, test_beta, np.pi))
    crit = []
    crit0 = []
    for bt in Beta:
        y = projection_b(FX, T, alpha, bt)
        y0 = projection_b(FX0, T, alpha, bt)
        crit.append(y)
        crit0.append(y0)
    argmin = np.argmin(crit)
    beta_straighforward = Beta[argmin]
    if verbose:
        fig = plt.figure(figsize=(7, 5))
        plt.plot(Beta, crit)
        plt.plot(Beta, crit0)
        plt.grid()
        plt.legend(['Empirical', 'Theoretical'])
        plt.xlabel('beta')
        plt.ylabel(f'Error(beta, {alpha})')
        plt.title('beta_real = '+str(test_beta)+', beta_est = '+str(round(beta_straighforward, 4))+f', alpha = {truncate_string(alpha)}, N = {N}')
        plt.axvline(x=beta_straighforward, color='green', linestyle='--')
        plt.axvline(x=test_beta, color='red', linestyle='--')
        print('beta = '+str(round(test_beta, 4))+', beta_straightforward_search = '+str(round(beta_straighforward, 4)))
    return beta_straighforward

def beta_estimate_error_functions_vs_scipy(T, alpha, test_beta, N):
    X = stable_sample(alpha, test_beta, N)
    Beta = list(np.arange(-1, 1, 0.01))+[]
    FX = F_ab(X, T)
    FX0 = []
    for t in T:
        FX0.append(psi(t, alpha, test_beta, np.pi))
    crit = []
    crit0 = []
    for bt in Beta:
        y = projection_b(FX, T, alpha, bt)
        y0 = projection_b(FX0, T, alpha, bt)
        crit.append(y)
        crit0.append(y0)
    argmin = np.argmin(crit)
    beta_straighforward = Beta[argmin]
    alpha_est, beta_scp, sgm, mu = levy_stable.fit(X, floc=0, fscale=1)
    print(f'alpha = {alpha}, beta = {test_beta}, ests:  beta_mlp = {truncate_string(beta_straighforward)}, beta_scp = {truncate_string(beta_scp)}')
    #return beta_straighforward


def beta_visualize_error_functions(TT, alpha, test_beta):
    Beta = list(np.arange(-1, 1, 0.01))+[]
    LEGEND = []
    j = 0
    plt.figure()
    for T in TT:
        FX0 = []
        for t in T:
            FX0.append(psi(t, alpha, test_beta, np.pi))

        crit0 = []
        for bt in Beta:

            y0 = projection_b(FX0, T, alpha, bt)

            crit0.append(y0)
        crit0 = np.array(crit0)/max(crit0)
        plt.plot(Beta, crit0)
        LEGEND.append(f'T[{j+1}]')
        j += 1
    plt.grid()
    plt.legend(LEGEND)
    plt.xlabel('beta')
    plt.ylabel('Error(beta)')
    plt.title('Visualization of different normalized error functions')
    plt.axvline(x=test_beta, color='red', linestyle='--')



def beta_visualize_sum_error_functions(TT, alpha, test_beta):
    Beta = list(np.arange(-1, 1, 0.01))+[]
    j = 0
    plt.figure()
    for T in TT:
        FX0 = []
        for t in T:
            FX0.append(psi(t, alpha, test_beta, np.pi))

        crit0 = []
        for bt in Beta:

            y0 = projection_b(FX0, T, alpha, bt)

            crit0.append(y0)
        crit0 = np.array(crit0)/max(crit0)
        if j == 0:
            crit_fin = crit0
        else:
            crit_fin += crit0
        j += 1
    plt.plot(Beta, crit_fin)
    plt.grid()
    plt.xlabel('beta')
    plt.ylabel('Error(beta)')
    plt.title(f'Visualization of different normalized error functions for alpha = {alpha}')
    plt.axvline(x=test_beta, color='red', linestyle='--')


def beta_meta_estimate(TT, alpha, X):
    Beta = list(np.arange(-1, 1, 0.01))+[]
    j = 0
    for T in TT:
        FX0 = F_ab(X, T)

        crit0 = []
        for bt in Beta:

            y0 = projection_b(FX0, T, alpha, bt)

            crit0.append(y0)
        crit0 = np.array(crit0)/max(crit0)
        if j == 0:
            crit_fin = crit0
        else:
            crit_fin += crit0
        j += 1
    argmin = np.argmin(crit_fin)
    beta_straighforward = Beta[argmin]
    return beta_straighforward


def beta_meta_estimate_vs_scipy(TT, alpha, X):
    Beta = list(np.arange(-1, 1, 0.01))+[]
    j = 0
    for T in TT:
        FX0 = F_ab(X, T)

        crit0 = []
        for bt in Beta:

            y0 = projection_b(FX0, T, alpha, bt)

            crit0.append(y0)
        crit0 = np.array(crit0)/max(crit0)
        if j == 0:
            crit_fin = crit0
        else:
            crit_fin += crit0
        j += 1
    argmin = np.argmin(crit_fin)
    beta_straighforward = Beta[argmin]
    alpha_est, beta_scp, sgm, mu = levy_stable.fit(X, floc=0, fscale=1)
    print(f'alpha = {alpha}, scipy_alpha_est = {truncate_string(alpha_est)}, beta_scipy = {truncate_string(beta_scp)}, beta_mlp = {truncate_string(beta_straighforward)}')