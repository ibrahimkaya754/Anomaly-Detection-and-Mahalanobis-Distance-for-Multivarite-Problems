from import_modules import *
from helper_functions import *

class prepare_inputs:
    def __init__(self,list_x_train_test_valid_sc, list_y_train_test_valid_sc, list_tests, list_params,
                 feature_keys, target_keys, list_x_flight_sc, list_y_flight_sc,cnn=False):
        
        ##########################################################################################################
        self.list_tests                 = list_tests
        self.list_params                = list_params
        self.feature_keys               = feature_keys
        self.target_keys                = target_keys
        self.list_x_train_test_valid_sc = list_x_train_test_valid_sc
        self.list_y_train_test_valid_sc = list_y_train_test_valid_sc
        self.list_x_flight_sc           = list_x_flight_sc
        self.list_y_flight_sc           = list_y_flight_sc
        self.dict_x_sc                  = {}
        self.dict_y_sc                  = {}
        self.cnn                        = cnn

        list_split_through = ['train','test','valid']

        for value,key in enumerate(list_split_through):
            self.dict_x_sc[key] = list_x_train_test_valid_sc[value]

        for value,key in enumerate(list_split_through):
            self.dict_y_sc[key] = list_y_train_test_valid_sc[value]

        print(' Shapes of the Train, Test and Validation Sets of Features \n')
        for ii in list_split_through:
            print('--------------------------------------------------------------')
            key_array = [jj for jj in self.dict_x_sc[ii].keys()]
            for kk in key_array:
                print(ii,'input set shape for',kk,'is: ', self.dict_x_sc[ii][kk].shape)
            
        print('\n*******************************************************************************\n')
        # Shapes of the Train, Test and Validation Sets of Output Targets
        print(' Shapes of the Train, Test and Validation Sets of Output Targets \n')
        for ii in list_split_through:
            print('--------------------------------------------------------------')
            for kk in self.target_keys:
                print(ii,'target output set shape for',kk,'is: ', self.dict_y_sc[ii][kk].shape)

        print('\n')        
        self.dict_x_flight_sc = {}
        self.dict_y_flight_sc = {}
        for test in self.list_tests[1:]:
            self.dict_x_flight_sc[test] = self.list_x_flight_sc[test][0]
            self.dict_y_flight_sc[test] = self.list_y_flight_sc[test][0]

        ##########################################################################################################

        self.dict_input_features = {}
        self.inp                 = {}
        self.inp_test            = {}
        self.inp_valid           = {}
        self.out                 = {}
        self.out_custom          = {}
        self.input_dl_train      = {}
        self.input_dl_test       = {}
        self.input_dl_valid      = {}
        self.input_dl_flight     = {}
        self.input_dl_origin     = {}
        self.out_dl_train        = {}
        self.out_dl_test         = {}
        self.out_dl_valid        = {}
        self.out_dl_flight       = {}
        self.out_dl_origin       = {}
        self.key_array           = [jj for jj in self.dict_x_sc['train'].keys()]

        for ii in key_array:
            self.dict_input_features[ii] = Input(shape=(self.dict_x_sc['train'][ii].shape[1],), name=ii)
            self.inp[ii]                 = self.dict_x_sc['train'][ii]  
            self.inp_test[ii]            = self.dict_x_sc['test'][ii]
            self.inp_valid[ii]           = self.dict_x_sc['valid'][ii]

        for ii in self.target_keys:
            self.out[ii+'_out']                 = self.dict_y_sc['train'][ii]
            self.out_custom[ii+'_custom_out']   = self.dict_y_sc['train'][ii]

        self.input_dl_train['input_all']  = self.dict_x_sc['train']['param1']
        self.input_dl_test['input_all']   = self.dict_x_sc['test']['param1']
        self.input_dl_valid['input_all']  = self.dict_x_sc['valid']['param1']
        self.out_dl_train['all_targets']  = self.dict_y_sc['train']['param1']
        self.out_dl_test['all_targets']   = self.dict_y_sc['test']['param1']
        self.out_dl_valid['all_targets']  = self.dict_y_sc['valid']['param1']

        for trgt in self.feature_keys[1:]:
            self.input_dl_train['input_all']  =  np.hstack((self.input_dl_train['input_all'],self.dict_x_sc['train'][trgt]))
            self.input_dl_test['input_all']   =  np.hstack((self.input_dl_test['input_all'],self.dict_x_sc['test'][trgt]))
            self.input_dl_valid['input_all']  =  np.hstack((self.input_dl_valid['input_all'],self.dict_x_sc['valid'][trgt]))
            
        for trgt in self.target_keys[1:]:
            self.out_dl_train['all_targets']  =  np.hstack((self.out_dl_train['all_targets'],self.dict_y_sc['train'][trgt]))
            self.out_dl_test['all_targets']   =  np.hstack((self.out_dl_test['all_targets'],self.dict_y_sc['test'][trgt]))
            self.out_dl_valid['all_targets']  =  np.hstack((self.out_dl_valid['all_targets'],self.dict_y_sc['valid'][trgt]))

        for test in self.list_tests[1:]:
            self.input_dl_flight[test] = {}
            self.out_dl_flight[test]   = {}

        for test in self.list_tests[1:]:
            self.input_dl_flight[test]['input_all'] = self.dict_x_flight_sc[test]['param1']
            self.out_dl_flight[test]['all_targets'] = self.dict_y_flight_sc[test]['param1']    

        for trgt in self.feature_keys[1:]:
            for test in self.list_tests[1:]:
                self.input_dl_flight[test]['input_all'] =  np.hstack((self.input_dl_flight[test]['input_all'],self.dict_x_flight_sc[test][trgt]))
                self.out_dl_flight[test]['all_targets'] =  np.hstack((self.out_dl_flight[test]['all_targets'],self.dict_y_flight_sc[test][trgt]))

        if self.cnn:
            self.input_all                     = Input(shape=(self.input_dl_train['input_all'].shape[1],1), name='input_all')
            self.input_dl_train['input_all']   = self.input_dl_train['input_all'].reshape(self.input_dl_train['input_all'].shape[0],
                                                                                          self.input_dl_train['input_all'].shape[1],1)
            self.input_dl_test['input_all']    = self.input_dl_test['input_all'].reshape(self.input_dl_test['input_all'].shape[0],
                                                                                         self.input_dl_test['input_all'].shape[1],1)
            self.input_dl_valid['input_all']   = self.input_dl_valid['input_all'].reshape(self.input_dl_valid['input_all'].shape[0],
                                                                                          self.input_dl_valid['input_all'].shape[1],1)
            
        else:
            self.input_all        = Input(shape=(self.input_dl_train['input_all'].shape[1],), name='input_all')

# BUILD CUSTOM NETWORK
class model(prepare_inputs):
    def __init__(self, list_x_train_test_valid_sc, list_y_train_test_valid_sc, list_tests, list_params,
                 feature_keys, target_keys, list_x_flight_sc, list_y_flight_sc,cnn,
                 mdl_name, act='tanh', trainable_layer=True, bottleneck=3, initializer='glorot_normal',
                 list_nn=[150,100,20],load_weights=True,scalers=None):
        super().__init__(list_x_train_test_valid_sc, list_y_train_test_valid_sc, list_tests, list_params,
                         feature_keys, target_keys, list_x_flight_sc, list_y_flight_sc,cnn=False)
        self.model_name      = mdl_name
        self.act             = act
        self.trainable_layer = trainable_layer
        self.init            = initializer
        self.opt             = Yogi(lr=0.001)
        self.list_nn         = list_nn
        self.bottleneck      = bottleneck
        self.losses          = {}
        self.lossWeights     = {}
        self.scaler_path     = {'feature' : None,
                                'target'  : None}
        self.regularization_paramater = 0.0
        self.dict_scalery    = scalers['scaler_y']
        self.dict_scalerx    = scalers['scaler_x']

        L1 = Dense(self.list_nn[0], activation=self.act,
                     kernel_initializer=self.init, trainable = self.trainable_layer,
                     kernel_regularizer=regularizers.l2(self.regularization_paramater))(self.input_all)

        for ii in range(1,len(self.list_nn)):
            L1 = Dense(self.list_nn[ii], activation=self.act, trainable = self.trainable_layer,
                         kernel_initializer=self.init,
                         kernel_regularizer=regularizers.l2(self.regularization_paramater))(L1)    

        L1 = Dense(self.bottleneck, activation='linear', name='bottleneck',
                         kernel_initializer=self.init,
                         kernel_regularizer=regularizers.l2(self.regularization_paramater))(L1)
        
        for ii in range(0,len(self.list_nn)):
            L1 = Dense(self.list_nn[-ii-1], activation=self.act, trainable = self.trainable_layer,
                         kernel_initializer=self.init,
                         kernel_regularizer=regularizers.l2(self.regularization_paramater))(L1)

        LOut = Dense(len(self.target_keys), activation=self.act, name='all_targets',
                         kernel_initializer=self.init,
                         kernel_regularizer=regularizers.l2(self.regularization_paramater))(L1)

        self.model                         = Model(inputs=[self.input_all], outputs=LOut)
        self.description                   = None
        self.losses['all_targets']         = huber_loss
        self.lossWeights['all_targets']    = 1.0
        self.model_path                    = os.getcwd()+"/" + self.model_name + '.hdf5'
        self.learning_rate_decrease_factor = 0.97
        self.learning_rate_patience        = 5
        self.number_of_params              = self.model.count_params()
        self.reduce_lr                     = ReduceLROnPlateau(monitor='val_loss', 
                                                               factor=self.learning_rate_decrease_factor,
                                                               patience=self.learning_rate_patience, 
                                                               min_lr=0.0000001, mode='min', verbose=1)
        self.checkpoint                    = ModelCheckpoint(self.model_path, 
                                                             monitor='val_loss', verbose=1, 
                                                             save_best_only=True, period=1, 
                                                             mode='min',save_weights_only=False)
        self.model.compile(optimizer=self.opt, loss=self.losses['all_targets'], metrics=['mse'])
        plot_model(self.model,to_file=self.model_name+'.png', show_layer_names=True,show_shapes=True)
        print('\n%s with %s params created' % (self.model_name,self.number_of_params))
        if os.path.exists(self.model_path):
            if load_weights:
                print('weights loaded for %s' % (self.model_name))
                self.model.load_weights(self.model_path)
 
        # Make the prediction for bottleneck layer
        self.bottleneck_layer = Model(self.model.input,self.model.get_layer('bottleneck').output)

        self.target_bn = ['dim'+str(ii) for ii in range(self.bottleneck)]
        
    def __describe__(self):
        return self.description
     
    def summary(self):
        self.model.summary()
        print('\nModel Name is: ',self.model_name)
        print('\nModel Path is: ',self.model_path)
        print('\nActivation Function is: ',self.act)
        print('\nLearning Rate Decreases by a factor of %s with patience of %s' % (self.learning_rate_decrease_factor,
                                                                                   self.learning_rate_patience))
        if self.description != None:
            print('\nModel Description: '+self.__describe__())
        
    def run(self,num_epochs,batch_size):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        print('Start Running \n')
        self.history = self.model.fit(self.input_dl_train,
                                      self.out_dl_train, 
                                      batch_size=self.batch_size, epochs=self.num_epochs, shuffle=True,
                                      callbacks=[self.checkpoint, self.reduce_lr],
                                      validation_data=(self.input_dl_test,self.out_dl_test), verbose=1)
        self.val_loss = np.min(self.history.history['val_loss'])
        
    def results(self,load_weights=False):
    
        if load_weights:
            self.model.load_weights(self.model_path)
            print('Weights Loaded')
        
        self.dict_y_flight    = {}
        self.dict_y_flight_bn = {}
        for test in self.list_tests[1:]:
            self.dict_y_flight[test] = {}
            self.dict_y_flight[test] = {}
        
        if self.cnn:
            self.input_dl_flight['input_all'] = self.input_dl_flight[test]['input_all'].reshape(self.input_dl_flight[test]['input_all'].shape[0],
                                                                                self.input_dl_flight[test]['input_all'].shape[1],1)
        
        # Make the prediction for training set
        self.out_dl_predicted_train = self.model.predict(self.input_dl_train, batch_size=None)
        print('Prediction for Training Set is Completed')

        # Make the prediction for test set
        self.out_dl_predicted_test = self.model.predict(self.input_dl_test, batch_size=None)
        print('Prediction for Test Set is Completed')

        # Make the prediction for validation set
        self.out_dl_predicted_valid = self.model.predict(self.input_dl_valid, batch_size=None)
        print('Prediction for Validation Set is Completed')
        
        # Make the prediction for flight set
        self.out_dl_predicted_flight    = {}
        self.out_dl_predicted_flight_bn = {}
        for test in self.list_tests[1:]:
            self.out_dl_predicted_flight[test]    = self.model.predict(self.input_dl_flight[test], batch_size=None)
            self.out_dl_predicted_flight_bn[test] = self.bottleneck_layer.predict(self.input_dl_flight[test], batch_size=None)
        print('Prediction for Flight Set is Completed')

        print('-------------------------------------------------------------------------------------')
        self.output_dl_train      = {}
        self.output_dl_test       = {}
        self.output_dl_valid      = {}
        self.output_dl_inv_test   = {}
        self.output_dl_inv_valid  = {}
        self.output_dl_inv_train  = {}
        self.dict_x_train         = {}
        self.dict_x_test          = {}
        self.dict_x_valid         = {}
        self.dict_y_train         = {}
        self.dict_y_test          = {}
        self.dict_y_valid         = {}
        self.output_dl_train_test_validation_origin     = {}
        self.output_dl_inv_train_test_validation_origin = {}

        for ii in range(len(self.target_keys)):
            self.output_dl_train[self.target_keys[ii]]  = self.out_dl_predicted_train[:,ii]
            self.output_dl_test[self.target_keys[ii]]   = self.out_dl_predicted_test[:,ii]
            self.output_dl_valid[self.target_keys[ii]]  = self.out_dl_predicted_valid[:,ii]


        self.output_dl_train_test_validation_origin['train']         = self.output_dl_train
        self.output_dl_train_test_validation_origin['test']          = self.output_dl_test
        self.output_dl_train_test_validation_origin['valid']         = self.output_dl_valid

        self.output_dl_inv_train_test_validation_origin['train']     = self.output_dl_inv_train
        self.output_dl_inv_train_test_validation_origin['test']      = self.output_dl_inv_test
        self.output_dl_inv_train_test_validation_origin['valid']     = self.output_dl_inv_valid
        
        self.output_dl_flight    = {}
        self.output_dl_flight_bn = {}
        self.output_dl_inv_flight = {}
        for test in self.list_tests[1:]:
            self.output_dl_flight[test]     = {}
            self.output_dl_flight_bn[test]  = {}
            self.output_dl_inv_flight[test] = {}
            
        for test in self.list_tests[1:]:
            for ii in range(len(self.target_keys)):
                self.output_dl_flight[test][self.target_keys[ii]] = self.out_dl_predicted_flight[test][:,ii]
            for ii in range(len(self.target_bn)):
                self.output_dl_flight_bn[test][self.target_bn[ii]] = self.out_dl_predicted_flight_bn[test][:,ii]
            for ii in self.target_keys:
                self.output_dl_inv_flight[test][ii] = self.dict_scalery[ii].inverse_transform(self.output_dl_flight[test][ii].reshape(-1,1))
                self.dict_y_flight[test][ii]        = self.dict_scalery[ii].inverse_transform(self.dict_y_flight_sc[test][ii].reshape(-1,1))

        print('\n\nMAE and Explained Variance Calculation for Train Set')
        print('-------------------------------------------------------------------------------------')
        for ii in self.target_keys:
            print(ii+" explained Variance of training set:", explained_variance_score(self.dict_y_sc['train'][ii], self.output_dl_train[ii].reshape(self.output_dl_train[ii].shape[0],1)))

        print('\n\nMAE and Explained Variance Calculation for Test Set')
        print('-------------------------------------------------------------------------------------')
        for ii in self.target_keys:
            print(ii+" explained Variance of test set:", explained_variance_score(self.dict_y_sc['test'][ii], self.output_dl_test[ii].reshape(self.output_dl_test[ii].shape[0],1)))

        print('\n\nMAE and Explained Variance Calculation for Validation Set')
        print('-------------------------------------------------------------------------------------')
        for ii in self.target_keys:
            print(ii+" explained Variance of validation set:", explained_variance_score(self.dict_y_sc['valid'][ii], self.output_dl_valid[ii].reshape(self.output_dl_valid[ii].shape[0],1)))
    
        # MAE and Explained Variance Calculations for Flight Sets - DL
        print('\n\nMAE and Explained Variance Calculation for Flight Sets')
        print('-------------------------------------------------------------------------------------')
        for test in self.list_tests[1:]:
            print('******************** ',test,' *****************************')
            for ii in self.target_keys:
                print(ii+" explained Variance of flight set:", explained_variance_score(self.dict_y_flight_sc[test][ii],
                                                                                        self.output_dl_flight[test][ii].reshape(self.output_dl_flight[test][ii].shape[0],1)))
        
        self.out_dl_predicted_train_bn  = self.bottleneck_layer.predict(self.input_dl_train, batch_size=None)
        self.out_dl_predicted_test_bn   = self.bottleneck_layer.predict(self.input_dl_test, batch_size=None)
        self.out_dl_predicted_valid_bn  = self.bottleneck_layer.predict(self.input_dl_valid, batch_size=None)

        self.output_dl_train_bn      = {}
        self.output_dl_test_bn       = {}
        self.output_dl_valid_bn      = {}
        self.output_dl_inv_test_bn   = {}
        self.output_dl_inv_valid_bn  = {}
        self.output_dl_inv_train_bn  = {}
        self.output_dl_train_test_validation_bn     = {}
        self.output_dl_inv_train_test_validation_bn = {}

        for ii in range(len(self.target_bn)):
            self.output_dl_train_bn[self.target_bn[ii]]  = self.out_dl_predicted_train_bn[:,ii]
            self.output_dl_test_bn[self.target_bn[ii]]   = self.out_dl_predicted_test_bn[:,ii]
            self.output_dl_valid_bn[self.target_bn[ii]]  = self.out_dl_predicted_valid_bn[:,ii]

        self.output_dl_train_test_validation_bn['train']    = self.output_dl_train_bn
        self.output_dl_train_test_validation_bn['test']     = self.output_dl_test_bn
        self.output_dl_train_test_validation_bn['valid']    = self.output_dl_valid_bn
        
        self.output_dl_train_test_valid = {}
        for target in self.list_params:
            self.output_dl_train_test_valid[target] = np.hstack((self.output_dl_train[target],self.output_dl_test[target],self.output_dl_valid[target]))


        self.output_dl_train_test_valid_bn = {}
        for target in self.target_bn:
            self.output_dl_train_test_valid_bn[target] = np.hstack((self.output_dl_train_bn[target],self.output_dl_test_bn[target],self.output_dl_valid_bn[target]))

    def plots(self,pnt_number=250,plot_train_test_valid=False):
        self.pnt_number = pnt_number
        if plot_train_test_valid:
            print('************ PLOTS FOR TRAIN SET ****************\n')
            for ii in range(len(self.target_keys)):
                print(self.target_keys[ii])
                mae = str(mean_absolute_error(self.output_dl_train[self.target_keys[ii]][:],(self.dict_y_sc['train'][self.target_keys[ii]])))
                plt.figure(figsize=(26,9))
                plt.plot(self.output_dl_train[self.target_keys[ii]][0:self.pnt_number], '--', markersize=1, label='Predicted', color = 'tab:red')
                plt.plot((self.dict_y_sc['train'][self.target_keys[ii]][0:self.pnt_number,0]), '--', markersize=3, label='Actual', color = 'tab:blue')
                plt.legend()
                plt.xlabel('Sample Point')
                plt.ylabel(self.target_keys[ii])
                plt.title('Mae for '+self.target_keys[ii]+' Prediction for Training Set: ' + mae)
                plt.grid()
                plt.show()

            print('************ PLOTS FOR TEST SET ****************\n')
            for ii in range(len(self.target_keys)):
                print(self.target_keys[ii])
                mae = str(mean_absolute_error(self.output_dl_test[self.target_keys[ii]][:],(self.dict_y_sc['test'][self.target_keys[ii]])))
                plt.figure(figsize=(26,9))
                plt.plot(self.output_dl_test[self.target_keys[ii]][0:self.pnt_number], '--', markersize=1, label='Predicted', color = 'tab:red')
                plt.plot((self.dict_y_sc['test'][self.target_keys[ii]][0:self.pnt_number,0]), '--', markersize=3, label='Actual', color = 'tab:blue')
                plt.legend()
                plt.xlabel('Sample Point')
                plt.ylabel(self.target_keys[ii])
                plt.title('Mae for '+self.target_keys[ii]+' Prediction for Test Set: ' + mae)
                plt.grid()
                plt.show()

            print('************ PLOTS FOR VALIDATION SET ****************\n')
            for ii in range(len(self.target_keys)):
                print(self.target_keys[ii])
                mae = str(mean_absolute_error(self.output_dl_valid[self.target_keys[ii]][:],(self.dict_y_sc['valid'][self.target_keys[ii]])))
                plt.figure(figsize=(26,9))
                plt.plot(self.output_dl_valid[self.target_keys[ii]][0:self.pnt_number], '--', markersize=1, label='Predicted', color = 'tab:red')
                plt.plot((self.dict_y_sc['valid'][self.target_keys[ii]][0:self.pnt_number,0]), '--', markersize=3, label='Actual', color = 'tab:blue')
                plt.legend()
                plt.xlabel('Sample Point')
                plt.ylabel(self.target_keys[ii])
                plt.title('Mae for '+self.target_keys[ii]+' Prediction for Valid Set: ' + mae)
                plt.grid()
                plt.show()
            
        print('************ PLOTS FOR FLIGHT SET ****************\n')
        for test in self.list_tests[1:]:
            print(' ------------ ',test, ' --------------------')
            for ii in range(len(self.target_keys)):
                print(self.target_keys[ii])
                mae = str(mean_absolute_error(self.output_dl_flight[test][self.target_keys[ii]][:],(self.dict_y_flight_sc[test][self.target_keys[ii]])))
                plt.figure(figsize=(26,9))
                plt.plot(self.output_dl_flight[test][self.target_keys[ii]][0:self.pnt_number], '--', markersize=1, label='Predicted', color = 'tab:red')
                plt.plot((self.dict_y_flight_sc[test][self.target_keys[ii]][0:self.pnt_number,0]), '--', markersize=3, label='Actual', color = 'tab:blue')
                plt.legend()
                plt.xlabel('Sample Point')
                plt.ylabel(self.target_keys[ii])
                plt.title('Mae for '+self.target_keys[ii]+' Prediction for Flight Set: ' + mae)
                plt.grid()
                plt.show()
        
    def scatter_plot_for_bottleneck(self):
        print('************ Scatter Plot for the BottleNeck Layer ****************\n')
        for test in self.list_tests[1:]:
            print('********** Scatter Plot for ',test, ' *********************************\n')
            cntr = 1
            for ii in range(len(self.target_bn)):
                pnt_number = 180000
                if ii == len(self.target_bn)-1:
                    break
                for jj in range(ii+1,len(self.target_bn)):
                    fig = plt.figure(figsize=(26,9))
                    plt.scatter(self.output_dl_flight_bn[test][self.target_bn[ii]][0:],self.output_dl_flight_bn[test][self.target_bn[jj]][0:],label='Flight')
                    plt.scatter(self.output_dl_train_bn[self.target_bn[ii]][0:],self.output_dl_train_bn[self.target_bn[jj]][0:],label='Train')
                    plt.scatter(self.output_dl_test_bn[self.target_bn[ii]][0:],self.output_dl_test_bn[self.target_bn[jj]][0:],label='Test')
                    plt.scatter(self.output_dl_valid_bn[self.target_bn[ii]][0:],self.output_dl_valid_bn[self.target_bn[jj]][0:],label='Valid')
                    plt.legend()
                    plt.xlabel(self.target_bn[ii])
                    plt.ylabel(self.target_bn[jj])
                    plt.grid()
                    #plt.xlim((-5.0,+5.0))
                   # plt.ylim((-10.0,+10.0))
                    plt.show()
                    fig.savefig('./images/bottleneck_'+str(cntr))
                    cntr = cntr + 1
        
    def histogram_for_bottleneck(self):
        for test in self.list_tests[1:]:
            print('*************** ',test, ' *******************************************\n')
            print('************ Histogram Plot for the BottleNeck Layer ****************\n')
            for ii in range(len(self.target_bn)):
                fig = plt.figure(figsize=(26,9))
                plt.hist(self.output_dl_train_bn[self.target_bn[ii]][0:],label='Trained',bins=100,color='tab:blue')
                plt.hist(self.output_dl_flight_bn[test][self.target_bn[ii]][0:],label='Flight',bins=100,color='tab:red')
                plt.legend()
                plt.xlabel(self.target_bn[ii])
                plt.grid()
                plt.xlim((-1.0,+1.0))
                plt.show()
                fig.savefig('./images/bottleneck_hist_'+str(ii))
                print("*******************************************************************************************************************")
                print("*******************************************************************************************************************")
            
    def mae(self):
        # Error for Training Data
        self.mae_for_training = {}
        for ii in self.target_keys:
            self.mae_for_training[ii] = np.zeros((self.output_dl_train[ii].shape[0],1))

        for ii in range(len(self.target_keys)):
            for jj in range(len(self.output_dl_train[self.target_keys[ii]])):
                self.mae_for_training[self.target_keys[ii]][jj,0] = self.output_dl_train[self.target_keys[ii]][jj]- self.dict_y_sc['train'][self.target_keys[ii]][jj,0]

        # Error for Test Data
        self.mae_for_test = {}
        for ii in self.target_keys:
            self.mae_for_test[ii] = np.zeros((self.output_dl_test[ii].shape[0],1))

        for ii in range(len(self.target_keys)):
            for jj in range(len(self.output_dl_test[self.target_keys[ii]])):
                self.mae_for_test[self.target_keys[ii]][jj,0] = self.output_dl_test[self.target_keys[ii]][jj]- self.dict_y_sc['test'][self.target_keys[ii]][jj,0]

        # Error for Validation Data
        self.mae_for_valid = {}
        for ii in self.target_keys:
            self.mae_for_valid[ii] = np.zeros((self.output_dl_valid[ii].shape[0],1))

        for ii in range(len(self.target_keys)):
            for jj in range(len(self.output_dl_valid[self.target_keys[ii]])):
                self.mae_for_valid[self.target_keys[ii]][jj,0] = self.output_dl_valid[self.target_keys[ii]][jj]- self.dict_y_sc['valid'][self.target_keys[ii]][jj,0]

        # Error for Flight Data
        self.mae_for_flight = {}
        for test in self.list_tests[1:]:
            self.mae_for_flight[test] = {}
        
        for test in self.list_tests[1:]:
            for ii in self.target_keys:
                self.mae_for_flight[test][ii] = np.zeros((self.output_dl_flight[test][ii].shape[0],1))

            for ii in range(len(self.target_keys)):
                for jj in range(len(self.output_dl_flight[test][self.target_keys[ii]])):
                    self.mae_for_flight[test][self.target_keys[ii]][jj,0] = self.output_dl_flight[test][self.target_keys[ii]][jj]- self.dict_y_flight_sc[test][self.target_keys[ii]][jj,0]
                
    def histogram_mae(self):
        for test in self.list_tests[1:]:
            print('*************** ',test, ' *******************************************\n')
            print('************ Histogram Plot for Mae ****************\n')
            for ii in range(len(self.target_keys)):
                fig = plt.figure(figsize=(25,36))
                plt.subplot(411)
                plt.hist(self.mae_for_training[self.target_keys[ii]],label='Training',bins=500)
                plt.legend()
                plt.xlabel(self.target_keys[ii])
                plt.grid()
                plt.xlim((-0.50,+0.50))

                plt.subplot(412)
                plt.hist(self.mae_for_test[self.target_keys[ii]],label='Test',bins=500)
                plt.legend()
                plt.xlabel(self.target_keys[ii])
                plt.grid()
                plt.xlim((-0.50,+0.50))

                plt.subplot(413)
                plt.hist(self.mae_for_valid[self.target_keys[ii]],label='Validation',bins=500)
                plt.legend()
                plt.xlabel(self.target_keys[ii])
                plt.grid()
                plt.xlim((-0.50,+0.50))

                plt.subplot(414)
                plt.hist(self.mae_for_flight[test][self.target_keys[ii]],label='Flight',bins=500)
                plt.legend()
                plt.xlabel(self.target_keys[ii])
                plt.grid()
                plt.xlim((-0.50,+0.50))
                plt.show()
                fig.savefig('./images/error_hist'+self.target_keys[ii])
                print("*******************************************************************************************************************")
                print("*******************************************************************************************************************")
                
    def corr(self):
        # Pearson Correlation
        print('Pearson Correlation is Calculated for the features and targets at the bottleneck\n')
        
        self.results_bn       = {}
        for test in self.list_tests:
            self.results_bn[test] = {}
            
        for param in self.target_bn:
            self.results_bn['test1'][param] = self.output_dl_train_bn[param][0:]
            for test in self.list_tests[1:]:
                self.results_bn[test][param] = self.output_dl_flight_bn[test][param][0:]
            
        self.Covariance_all   = {}
        for param in self.target_bn:
            self.Covariance_all[param]  = {}   

        for param in self.target_bn:
            for test in self.list_tests:
                self.Covariance_all[param][test] = {}
    
        for param in self.target_bn:
            for test1 in self.list_tests:
                 for test2 in self.list_tests:
                    self.Covariance_all[param][test1][test2] = covar(self.results_bn[test1][param],self.results_bn[test2][param])
        
        self.sigma_all   = {}
        for param in self.target_bn:
            self.sigma_all[param]  = {}

        for param in self.target_bn:
            for test in self.list_tests:
                self.sigma_all[param][test] = sigma(self.results_bn[test][param])

        self.Corr = {}
        for param in self.target_bn:
            self.Corr[param]  = {}

        for param in self.target_bn:
            for test in self.list_tests:
                self.Corr[param][test] = {}

        for param in self.target_bn:
            for test1 in self.list_tests:
                for test2 in self.list_tests:
                    self.Corr[param][test1][test2] = self.Covariance_all[param][test1][test2] / (self.sigma_all[param][test1]*self.sigma_all[param][test2])
                    
        # Scaler Plot for the Correlations of Targets with Features
        for param in self.target_bn:
            print('\nCorrelation Coefficient for %s for different Test Data' % (param))
            plt.figure(figsize=(26,9))
            plt.scatter(np.arange(len(self.Corr[param]['test2'])-1),[self.Corr[param]['test2'][test] for test in self.list_tests[1:]],label=  param+'_vs_'+param)
            plt.legend()
            plt.xlabel([[test for test in self.list_tests[1:]]])
            plt.ylabel(param)
            plt.title('Correlation for %s obtained from the prediction of bottle neck of AutoEncoder' % ([test for test in self.list_tests[1:]]))
            plt.grid()
            plt.show()
              
    def mahal(self,u,v,str1,str2,scatter_plot=False,histogram_plot=False):
        for key in u.keys():
            mahal = mahalanobis(v[key].reshape(1,v[key].shape[0]),u[key].reshape(1,u[key].shape[0]))
            if scatter_plot:
                plt.figure(figsize=(26,9))
                plt.scatter(np.arange(mahal.diagonal().shape[0]),mahal.diagonal(),
                            label='Mahalanobis Distance of %s wrt %s' % (str1,str2))
                plt.legend()
                plt.xlabel(key)
                plt.grid()
                plt.show()
            if histogram_plot:
                plt.figure(figsize=(26,9))
                plt.hist(mahal.diagonal(),label='Mahalanobis Distance of %s wrt %s' % (str1,str2),
                        bins=100)
                plt.legend()
                plt.xlabel(key)
                plt.grid()
                plt.show()
#         fig.savefig('./images/mahalanobis_'+kk)
    
    def writeStandartScaler_AsMatFile(self,scaler,fileName,keys):
        if os.path.exists('./MatFiles/')==False:
            os.makedirs('./MatFiles/')
        self.mean      = {}
        self.variance  = {}
        self.scale     = {}
        self.scaler    = {}
        for key in keys:
            self.mean[key]      = scaler[key].mean_
            self.variance[key]  = scaler[key].var_
            self.scale[key]     = scaler[key].scale_
        self.scaler['mean']     = self.mean
        self.scaler['variance'] = self.variance
        self.scaler['scale']    = self.scale
        sio.savemat(fileName, self.scaler)
        return self.scaler
    
    def writeMinMaxScaler_AsMatFile(self,scaler,fileName,keys):
        if os.path.exists('./MatFiles/')==False:
            os.makedirs('./MatFiles/')
        self.min      = {}
        self.max      = {}
        self.scale    = {}
        self.data_min = {}
        self.data_max = {}
        self.scaler   = {}

        for key in keys:
            self.min[key], self.max[key] = scaler[key].feature_range
            self.scale[key]              = scaler[key].scale_
            self.data_min[key]           = scaler[key].data_min_
            self.data_max[key]           = scaler[key].data_max_
        self.scaler['min']      = self.min
        self.scaler['max']      = self.max
        self.scaler['scale']    = self.scale
        self.scaler['data_min'] = self.data_min
        self.scaler['data_max'] = self.data_max
        sio.savemat(fileName, self.scaler)
        return self.scaler
