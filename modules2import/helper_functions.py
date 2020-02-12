# Helper Functions

from import_modules import *

def huber_loss(desired,predicted):
    error = predicted - desired
    return K.mean(K.sqrt(1+K.square(error))-1,axis=-1)

def covar(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    Cov_numerator = sum(((a - x_mean)*(b - y_mean)) for a, b in zip(x, y))
    Cov_denomerator = len(x) - 1
    Covariance = (Cov_numerator / Cov_denomerator)
    return  Covariance

def covar2(x, y):
    X = np.column_stack([x, y])
    X -= X.mean(axis=0) 
    denomerator= len(X) - 1 
    return np.dot(X.T, X.conj()) / denomerator

def mahalanobis(x, y):
    # distance of every element of x according to y
    mean            = np.mean(y)
    x_minus_meant_T = (x - mean).T
    x_minus_mean    = x - mean
    Covariance      = covar(x.T, y.T)
    Covariance      = Covariance.reshape(1,1)
    inv_covmat      = np.linalg.inv(Covariance)
    left_term       = np.dot(x_minus_meant_T, inv_covmat)
    D_square        = np.dot(left_term, x_minus_mean)
    return np.sqrt(np.abs(D_square))

def sigma(x):
    x_mean = np.mean(x)
    sigma_numerator    = sum((a-x_mean)*(a-x_mean) for a in x)
    sigma_denomerator  = len(x) - 1
    sigma              = sigma_numerator / sigma_denomerator
    sigma              = np.sqrt(sigma)
    return sigma

def corr(data,parameters=False,datasets=False):
        list_tests_  = [key for key in data]
        list_params_ = [key for key in data[list_tests_[0]]]
        
        if datasets:
            # Pearson Correlation for the data sets. The same parameters are investigated
            print('Pearson Correlation is Calculated for the data sets (test1 to test5) for the same parameters')
            Covariance_all   = {}
            for param in list_params_:
                Covariance_all[param]  = {}   

            for param in list_params_:
                for test in list_tests_:
                    Covariance_all[param][test] = {}

            for param in list_params_:
                for test1 in list_tests_:
                    for test2 in list_tests_:
                        Covariance_all[param][test1][test2] = covar(data[test1][param],data[test2][param])

            sigma_all   = {}
            for param in list_params_:
                sigma_all[param]  = {}


            for param in list_params_:
                for test in list_tests_:
                    sigma_all[param][test] = sigma(data[test][param])

            Corr = {}
            for param in list_params_:
                Corr[param]  = {}

            for param in list_params_:
                for test in list_tests_:
                    Corr[param][test] = {}

            for param in list_params_:
                for test1 in list_tests_:
                    for test2 in list_tests_:
                        Corr[param][test1][test2] = Covariance_all[param][test1][test2] / (sigma_all[param][test1]*sigma_all[param][test2])

            # Scaler Plot for the Correlations of Targets with Features
            for param in list_params_:
                print('%s of %s vs %s of %s' % (param,list_tests_[0],param,list_tests_))
                plt.figure(figsize=(26,9))
                plt.scatter(np.arange(len(Corr[param][list_tests_[0]])),[Corr[param][list_tests_[0]][test] for test in list_tests_],label=  param+'_vs_'+param)
                plt.legend()
                plt.xlabel([list_tests_,list_tests_])
                plt.ylabel(param)
                plt.title('Correlation for the %s of %s' % (param,list_tests_[0]))
                plt.grid()
                plt.show()
            
            print('\n ************************************************************************************************** \n')
        
        if parameters:
            # Pearson Correlation for the parameters. Different parameters of the same data set are investigated
            print('Pearson Correlation is Calculated for the different parameters of the same dataset\n')
            
            Covariance_all   = {}
            for test in list_tests_:
                Covariance_all[test]  = {}

            for test in list_tests_:
                for param in list_params_:
                    Covariance_all[test][param] = {}
                    
            for test in list_tests_:
                for param1 in list_params_:
                    for param2 in list_params_:
                        Covariance_all[test][param1][param2] = covar(data[test][param1],data[test][param2])

            sigma_all   = {}
            for test in list_tests_:
                sigma_all[test]  = {}

            for test in list_tests_:
                for param in list_params_:
                    sigma_all[test][param] = sigma(data[test][param])

            Corr = {}
            for test in list_tests_:
                Corr[test]  = {}

            for test in list_tests_:
                for param in list_params_:
                    Corr[test][param] = {}

            for test in list_tests_:
                for param1 in list_params_:
                    for param2 in list_params_:
                        Corr[test][param1][param2] = Covariance_all[test][param1][param2] / (sigma_all[test][param1]*sigma_all[test][param2])

            # Scaler Plot for the Correlations of Targets with Features
            for test in list_tests_:
                print('%s vs the other params for %s' % (list_params_[0],test))
                plt.figure(figsize=(26,9))
                plt.scatter(np.arange(len(Corr[test][list_params_[0]])),[Corr[test][list_params_[0]][param] for param in list_params_],label=  test+'_vs_'+test)
                plt.legend()
                plt.xlabel([list_params_,list_params_])
                plt.ylabel(list_params_[0])
                plt.title('Correlation for %s at the given dataset %s' % (list_params_[0],test))
                plt.grid()
                plt.show()
            
            print('\n ************************************************************************************************** \n')    
            
            for param1 in list_params_:
                print('%s vs the other params for %s' % (param1,list_tests_[0]))
                plt.figure(figsize=(26,9))
                plt.scatter(np.arange(len(Corr[list_tests_[0]][list_params_[0]])),[Corr[list_tests_[0]][param1][param2] for param2 in list_params_],label=  list_tests_[0]+'_vs_'+list_tests_[0])
                plt.legend()
                plt.xlabel([list_params_,list_params_])
                plt.ylabel(param1)
                plt.title('Correlation for the %s vs the other params' % (param1))
                plt.grid()
                plt.show()
            print('\n ************************************************************************************************** \n')
            
        return Covariance_all,sigma_all,Corr
