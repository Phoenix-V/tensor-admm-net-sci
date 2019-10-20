import tensorflow as tf
import numpy as np
import yaml
import os
import h5py
import time
import sys
import math

from Lib.Data_Processing import *
from Lib.Utility import *
from Model.Decoder_Model import Depth_Decoder
from Model.Base_Handler import Basement_Handler


class Decoder_Handler(Basement_Handler):
    def __init__(self, dataset_name, model_config, sess, is_training=True):
        
        # Initialization of Configuration, Parameter and Datasets
        super(Decoder_Handler, self).__init__(sess=sess, model_config=model_config, is_training=is_training)
        self.initial_parameter()
        self.data_assignment(dataset_name)

        # Data Generator
        #self.gen_train = Data_Generator_File(dataset_name,self.set_train,self.sense_mask,self.batch_size,is_training=True)
        #self.gen_valid = Data_Generator_File(dataset_name,self.set_valid,self.sense_mask,self.batch_size,is_training=False)
        self.gen_test  = Data_Generator_File(dataset_name,self.set_test,self.sense_mask,self.batch_size,is_training=False)
        
        # Define the general model and the corresponding input
        shape_meas = (self.batch_size,) + self.sense_mask.shape[:2] + (1,)
        shape_sense = self.sense_mask.shape
        shape_truth,shape_cross = (self.batch_size,)+shape_sense, shape_sense+(self.sense_mask.shape[-1],)
        print shape_meas,shape_sense,shape_truth,shape_cross
        
        self.meas_sample = tf.placeholder(tf.float32, shape=shape_meas, name='input_meas')
        self.initial_net = tf.placeholder(tf.float32, shape=shape_truth,name='input_init')
        self.sense_cross = tf.placeholder(tf.float32, shape=shape_cross,name='matrix_cross')
        self.sense_matrix = tf.placeholder(tf.float32, shape=shape_sense, name='input_mat')
        self.truth_sample = tf.placeholder(tf.float32, shape=shape_truth, name='output_truth')
        
        # Initialization for the model training procedure.
        self.learning_rate = tf.get_variable('learning_rate', shape=(), initializer=tf.constant_initializer(self.lr_init),
                                             trainable=False)
        self.lr_new = tf.placeholder(tf.float32, shape=(), name='lr_new')
        self.lr_update = tf.assign(self.learning_rate, self.lr_new, name='lr_update')
        self.train_test_valid_assignment()
        self.trainable_parameter_info()
        self.saver = tf.train.Saver(tf.global_variables())

    def initial_parameter(self):
        # Configuration Set
        config = self.model_config
        
        # Model Input Initialization
        self.batch_size = int(config.get('batch_size',1))
        self.upbound = float(config.get('upbound',1))
        
        # Initialization for Training Controler
        self.epochs = int(config.get('epochs',100))
        self.patience = int(config.get('patience',30))
        self.lr_init = float(config.get('learning_rate',0.001))
        self.lr_decay_coe = float(config.get('lr_decay',0.1))
        self.lr_decay_epoch = int(config.get('lr_decay_epoch',20))
        self.lr_decay_interval = int(config.get('lr_decay_interval',10))

    def data_assignment(self,dataset_name):
        # Division for train, test and validation
        model_config = self.model_config
        set_train, set_test, set_valid, self.sense_mask, sample = Data_Division(dataset_name)
        
        # The value of the position is normalized (the value of lat and lon are all limited in range(0,1))
        scalar = limit_scalar(self.sense_mask)
        self.phi_cross = scalar.phi_cross
        
        self.set_test, disp_test  = scalar.seperate_normalization(set_test)
        self.test_size  = int(np.ceil(float(disp_test[0]) /self.batch_size))
        
        #self.set_train,disp_train = scalar.overlap_normalization(set_train)
        #self.set_valid,disp_valid = scalar.overlap_normalization(set_valid)
        #self.train_size = int(np.ceil(float(disp_train[0])/self.batch_size))
        #self.valid_size = int(np.ceil(float(disp_valid[0])/self.batch_size))
        
    def train_test_valid_assignment(self):
        
        value_set = (self.meas_sample,self.initial_net,self.truth_sample,tf.expand_dims(self.sense_matrix,0),self.sense_cross)
        
        with tf.name_scope('Train'):
            with tf.variable_scope('Depth_Decoder', reuse=False):
                self.Decoder_train = Depth_Decoder(value_set,self.learning_rate,self.sess,self.model_config,is_training=True)
        with tf.name_scope('Val'):
            with tf.variable_scope('Depth_Decoder', reuse=True):
                self.Decoder_valid = Depth_Decoder(value_set,self.learning_rate,self.sess,self.model_config,is_training=False)
                
    def train(self):
        self.sess.run(tf.global_variables_initializer())
        print ('Training Started')
        if self.model_config.get('model_filename',None) is not None:
            self.restore()
            print 'Pretrained Model Downloaded'
        else:
            print 'New Model Training'
        epoch_cnt,wait,min_val_loss = 0,0,float('inf')
        
        while epoch_cnt <= self.epochs:
            
            # Training Preparation: Learning rate pre=setting, Model Interface summary.
            start_time = time.time()
            cur_lr = self.calculate_scheduled_lr(epoch_cnt)
            train_fetches = {'global_step': tf.train.get_or_create_global_step(), 
                             'train_op':self.Decoder_train.train_op,
                             'metrics':self.Decoder_train.metrics,
                             'pred_orig':self.Decoder_train.decoded_image,
                             'loss':self.Decoder_train.loss}
            valid_fetches = {'global_step': tf.train.get_or_create_global_step(),
                             'pred_orig':self.Decoder_valid.decoded_image,
                             'metrics':self.Decoder_valid.metrics,
                             'loss':self.Decoder_valid.loss}
            Tresults,Vresults = {"loss":[],"psnr":[],"ssim":[],"mse":[]},{"loss":[],"psnr":[],"ssim":[],"mse":[]}
            
            # Framework and Visualization SetUp for Training 
            for trained_batch in range(0,self.train_size):
                (measure_train,ground_train,netinit_train,_) = self.gen_train.next()
                feed_dict_train = {self.meas_sample:measure_train,
                                   self.truth_sample:ground_train,
                                   self.initial_net:netinit_train,
                                   self.sense_matrix:self.sense_mask,
                                   self.sense_cross:self.phi_cross}
                train_output = self.sess.run(train_fetches,feed_dict=feed_dict_train)
                Tresults["loss"].append(train_output['loss'])
                Tresults["psnr"].append(train_output['metrics'][0])
                Tresults["ssim"].append(train_output['metrics'][1])
                Tresults["mse"].append(train_output['metrics'][2])
                message = "Train Epoch [%2d/%2d] Batch [%d/%d] lr: %.4f, loss: %.8f psnr: %.4f" % (
                    epoch_cnt, self.epochs, trained_batch, self.train_size, cur_lr, Tresults["loss"][-1], Tresults["psnr"][-1])
                if trained_batch%10 == 0:
                    print message
                    
            # Framework and Visualization SetUp for Validation
            validation_time = []
            for valided_batch in range(0,self.valid_size):
                (measure_valid,ground_valid,netinit_valid,index_valid) = self.gen_valid.next()
                feed_dict_valid = {self.meas_sample:measure_valid,
                                   self.truth_sample:ground_valid,
                                   self.initial_net:netinit_valid,
                                   self.sense_matrix:self.sense_mask,
                                   self.sense_cross:self.phi_cross}
                start_time = time.time()
                valid_output = self.sess.run(valid_fetches,feed_dict=feed_dict_valid)
                end_time = time.time()
                validation_time.append(end_time-start_time)
                Vresults["loss"].append(valid_output['loss'])
                Vresults["psnr"].append(valid_output['metrics'][0])
                Vresults["ssim"].append(valid_output['metrics'][1])
                Vresults["mse"].append(valid_output['metrics'][2])
                message = "Valid Epoch [%2d/%2d] Batch [%d/%d] lr: %.4f, loss: %.8f psnr: %.4f" % (
                    epoch_cnt, self.epochs, valided_batch, self.valid_size, cur_lr, Vresults["loss"][-1], Vresults["psnr"][-1])
            print 'Validation Time:', validation_time
                    
            # Information Logging for Model Training and Validation (Maybe for Curve Plotting)
            Tloss,Vloss = np.mean(Tresults["loss"]),np.mean(Vresults["loss"])
            train_psnr,valid_psnr = np.mean(Tresults["psnr"]),np.mean(Vresults["psnr"])
            train_ssim,valid_ssim = np.mean(Tresults["ssim"]),np.mean(Vresults["ssim"])
            train_mse, valid_mse  = np.mean(Tresults["mse"]), np.mean(Vresults["mse"])
            summary_format = ['loss/train_loss','loss/valid_loss','metric/train_psnr','metric/train_ssim',
                              'metric/valid_psnr','metric/valid_ssim']
            summary_data = [Tloss, Vloss, train_psnr, train_ssim, valid_psnr, valid_ssim]
            self.summary_logging(train_output['global_step'], summary_format, summary_data)
            end_time = time.time()
            message = 'Epoch [%3d/%3d] Train(Valid) loss: %.4f(%.4f), T PSNR(MSE) %s(%s), V PSNR(MSE) %s(%s), time %s' % (
                epoch_cnt, self.epochs, Tloss, Vloss, train_psnr, train_mse, valid_psnr, valid_mse, np.mean(validation_time))
            self.logger.info(message)
            
            if epoch_cnt%10 == 0 or Vloss <= min_val_loss:
                matcont = {}
                matcont[u'truth'],matcont[u'pred'],matcont[u'meas'] = ground_valid,valid_output['pred_orig'],measure_valid
                hdf5storage.write(matcont, '.', self.log_dir+'/Data_Visualization_%d.mat' % (epoch_cnt), 
                                  store_python_metadata=False, matlab_compatible=True)
            if Vloss <= min_val_loss:
                model_filename = self.save_model(self.saver, epoch_cnt, Vloss)
                self.logger.info('Val loss decrease from %.4f to %.4f, saving to %s' % (min_val_loss, Vloss, model_filename))
                min_val_loss,wait = Vloss,0
            else:
                wait += 1
                if wait > self.patience:
                    model_filename = self.save_model(self.saver, epoch_cnt, Vloss)
                    self.logger.info('Val loss decrease from %.4f to %.4f, saving to %s' % (min_val_loss,Vloss,model_filename))                        
                    self.logger.warn('Early stopping at epoch: %d' % (epoch_cnt))
                    break
            
            epoch_cnt += 1
            sys.stdout.flush()

    def test(self):
        
        print "Testing Started"
        self.restore()
        
        test_fetches = {'pred_orig':   self.Decoder_valid.decoded_image}
        time_list = []
        for tested_batch in range(0,self.test_size):
            (measure_test,ground_test,netinit_test,index_test) = self.gen_test.next()
            print index_test
            feed_dict_test = {self.meas_sample:measure_test,
                              self.truth_sample:ground_test,
                              self.initial_net:netinit_test,
                              self.sense_matrix:self.sense_mask,
                              self.sense_cross:self.phi_cross}
            start_time = time.time()
            test_output = self.sess.run(test_fetches,feed_dict=feed_dict_test)
            end_time = time.time()
            time_list.append(end_time-start_time)
            message = "Test [%d/%d] time %s"%(tested_batch+1,self.test_size,time_list[-1])
            matcontent = {}
            matcontent[u'truth'],matcontent[u'pred'],matcontent[u'meas'] = ground_test,test_output['pred_orig'],measure_test
            hdf5storage.write(matcontent, '.', self.log_dir+'/Data_Visualization_%d.mat' % (tested_batch), 
                              store_python_metadata=False, matlab_compatible=True)
            print message
            
        
    def calculate_scheduled_lr(self, epoch, min_lr=1e-10):
        decay_factor = int(math.ceil((epoch - self.lr_decay_epoch)/float(self.lr_decay_interval)))
        new_lr = self.lr_init * (self.lr_decay_coe ** max(0, decay_factor))
        new_lr = max(min_lr, new_lr)
        
        self.logger.info('Current learning rate to: %.6f' % new_lr)
        sys.stdout.flush()
        
        self.sess.run(self.lr_update, feed_dict={self.lr_new: new_lr})
        self.Decoder_train.set_lr(self.learning_rate) 
        return new_lr
