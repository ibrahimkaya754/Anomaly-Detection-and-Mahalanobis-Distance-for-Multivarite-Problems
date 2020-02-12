# Helper Functions

from import_modules import *

def get_data_for_training(datF,target_output,features_input,sclr,shuffle=True):
    print('DATA SET PREPARATON FOR TRAINING, TESTING AND VALIDATION\n')
    print('\n****************************************************************************')
    number_of_data              = int(datF.describe()[datF.columns[0]]['count'])
    print('\nnumber of data     =',number_of_data)
    
#######################################################################################        
    ########## PREPARE FEATURE INPUTS ###############
    dict_x_origin               = {}
    
    for keys in features_input:
        dict_x_origin[keys] = datF[keys].values
        
    print('input_features     =',features_input)
    print('\n****************************************************************************')
    
#######################################################################################        
    ########## PREPARE TARGET OUTPUTS ###############
    dict_y_origin = {}
    for keys in target_output:
        dict_y_origin[keys] =  datF[keys].values

    print('target_output      =',target_output)
    print('\n****************************************************************************')
    
#######################################################################################
    ########### SPLIT THE WHOLE DATA TO TRAIN - TEST - VALIDATION SETS ###############
    ####### 1st Seperate the Validation data from the whole data set ######## 
    indices1          = np.arange(number_of_data)
    split1            = train_test_split(indices1, test_size = 0.10, random_state=1038, shuffle=shuffle)
    indices1_train_test, indices1_valid = split1
    dict_x_train_test = {}
    dict_x_valid      = {}
    dict_y_train_test = {}
    dict_y_valid      = {}
    print('\nSPLITTING THE WHOLE DATA TO TRAIN-TEST AND VALIDATION SETS')
    for key in features_input:
        dict_x_train_test[key] = dict_x_origin[key][indices1_train_test].reshape((-1,1))
        dict_x_valid[key]      = dict_x_origin[key][indices1_valid].reshape((-1,1))
    for key in target_output:
        dict_y_train_test[key] = dict_y_origin[key][indices1_train_test].reshape((-1,1))
        dict_y_valid[key]      = dict_y_origin[key][indices1_valid].reshape((-1,1))
    print('Done!')
    print('\n****************************************************************************')
    #########################################################################
    ####### 2nd Split the train_test data sets into train and test ##########
    split2       = train_test_split(indices1_train_test, test_size = 0.10, random_state=2761, shuffle=True)
    indices2_train, indices2_test = split2
    dict_x_train = {}
    dict_x_test  = {}
    dict_y_train = {}
    dict_y_test  = {}
    print('\nSPLITTING THE TRAIN-TEST DATA TO TRAIN AND TEST SETS')
    print('Done!')
    print('\n****************************************************************************')
    for key in features_input:
        dict_x_train[key] = dict_x_origin[key][indices2_train].reshape((-1,1))
        dict_x_test[key]  = dict_x_origin[key][indices2_test].reshape((-1,1))
    for key in target_output:
        dict_y_train[key] = dict_y_origin[key][indices2_train].reshape((-1,1))
        dict_y_test[key]  = dict_y_origin[key][indices2_test].reshape((-1,1))
    print('\nPRINTING TRAIN, TEST AND VALIDATION SETS')
    print('\n ***Input Features***') 
    for key in features_input:
        print('-----------------------------------------------')
        print('input features are       : ',key)
        print('train set shape for',key,'is          :' ,dict_x_train[key].shape)
        print('test set shape for',key,'is           :' ,dict_x_test[key].shape)
        print('validation set shape for',key,'is     :' ,dict_x_valid[key].shape)
    print('\n ***Output Targets***')    
    for key in target_output:
        print('-----------------------------------------------')
        print(key,'label shape for train_set is      :', dict_y_train[key].shape)
        print(key,'label shape for test_set is       :', dict_y_test[key].shape)
        print(key,'label shape for validation_set is :', dict_y_valid[key].shape)
    print('\n****************************************************************************')
# #######################################################################################        
    ###################### SCALING ###############################
    dict_scalerx    = {}
    dict_x_train_sc = {}
    dict_x_test_sc  = {}
    dict_x_valid_sc = {}
    for key in features_input:
        if sclr == 'minmax':
            scalerx = MinMaxScaler((-0.5,0.5))
        elif sclr == 'robust':
            scalerx = RobustScaler()
        elif sclr == 'standard':
            scalerx = StandardScaler()

        scx                  = scalerx.fit(dict_x_train[key])
        dict_x_train_sc[key] = scx.transform(dict_x_train[key])
        dict_x_test_sc[key]  = scx.transform(dict_x_test[key])
        dict_x_valid_sc[key] = scx.transform(dict_x_valid[key])
        dict_scalerx[key]    = scx
    list_x_train_test_valid_sc   = [dict_x_train_sc,dict_x_test_sc,dict_x_valid_sc]
    
    dict_scalery    = {}
    dict_y_train_sc = {}
    dict_y_test_sc  = {}
    dict_y_valid_sc = {}
    for key in target_output:
        if sclr == 'minmax':
            scalery = MinMaxScaler((-0.5,0.5))
        elif sclr == 'robust':
            scalery = RobustScaler()
        elif sclr == 'standard':
            scalery = StandardScaler()
        
        scy                  = scalery.fit(dict_y_train[key])
        dict_y_train_sc[key] = scy.transform(dict_y_train[key])
        dict_y_test_sc[key]  = scy.transform(dict_y_test[key])
        dict_y_valid_sc[key] = scy.transform(dict_y_valid[key])
        dict_scalery[key]    = scy
    list_y_train_test_valid_sc   = [dict_y_train_sc,dict_y_test_sc,dict_y_valid_sc]
#######################################################################################
    return list_x_train_test_valid_sc, list_y_train_test_valid_sc, dict_scalerx, dict_scalery


def get_data_for_testing(datF,target_output,features_input,dict_scalerx,dict_scalery):
    print('DATA SET PREPARATON FOR TRAINING, TESTING AND VALIDATION\n')
    print('\n****************************************************************************')
    number_of_data              = int(datF.describe()[datF.columns[0]]['count'])
    print('\nnumber of data     =',number_of_data)
    
#######################################################################################        
    ########## PREPARE FEATURE INPUTS ###############
    dict_x_origin               = {}
    
    for keys in features_input:
        dict_x_origin[keys] = datF[keys].values
        
    print('input_features     =',features_input)
    print('\n****************************************************************************')
    
#######################################################################################        
    ########## PREPARE TARGET OUTPUTS ###############
    dict_y_origin = {}
    for keys in target_output:
        dict_y_origin[keys] =  datF[keys].values

    print('target_output      =',target_output)
    print('\n****************************************************************************')
    
#######################################################################################
    ########### SPLIT THE WHOLE DATA TO TRAIN - TEST - VALIDATION SETS ###############
    ####### 1st Seperate the Validation data from the whole data set ######## 
    dict_x_flight     = {}
    dict_y_flight     = {}
    print('\nSPLITTING THE WHOLE DATA TO TRAIN-TEST AND VALIDATION SETS')
    for key in features_input:
        dict_x_flight[key] = dict_x_origin[key].reshape((-1,1))
    for key in target_output:
        dict_y_flight[key] = dict_y_origin[key].reshape((-1,1))
    print('Done!')
    print('\n****************************************************************************')
# #######################################################################################        
    ###################### SCALING ###############################
    dict_x_flight_sc = {}
    dict_y_flight_sc = {}
    for key in features_input:
        scx                  = dict_scalerx[key]
        dict_x_flight_sc[key] = scx.transform(dict_x_flight[key])
        
    list_x_flight_sc   = [dict_x_flight_sc]
    
    dict_y_flight_sc = {}
    for key in target_output:
        scy                   = dict_scalery[key]
        dict_y_flight_sc[key] = scy.transform(dict_y_flight[key])
 
    list_y_flight_sc   = [dict_y_flight_sc]
#######################################################################################
    return list_x_flight_sc, list_y_flight_sc
