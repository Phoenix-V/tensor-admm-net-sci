import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
import numpy as np
import math
import os
import json

from Lib.Utility import *
from Model.Base_TFModel import Basement_TFModel

class Depth_Decoder(Basement_TFModel):
    
    def __init__(self, value_sets, init_learning_rate, sess, config, is_training=True, *args, **kwargs):
        
        super(Depth_Decoder, self).__init__(sess=sess, config=config, learning_rate=init_learning_rate,is_training=is_training)
        '''
        Arguments:
            measurement: [batch, height, width, 1] compressed measurement
            initial_net: [batch, height, width, depth] phi_T_y 
            groundtruth: [batch, height, width, depth] 
            sense_mat: [1, height, width, depth] sensing matris phi 
            sense_cross: [height, width, depth, depth] phi_T_phi         
        '''
        (measurement,initial_net,groundtruth,sense_mat,sense_cross) = value_sets
        self.height,self.width,self.ratio = sense_mat.get_shape().as_list()[1:]
        
        # Initialization of the model hyperparameter, enc-dec structure, evaluation metric & Optimizier
        self.initial_parameter()
        #with tf.device("GPU:1"):
        self.decoded_image = self.encdec_handler(measurement,initial_net,groundtruth,sense_mat,sense_cross, 0.8)
        self.metric_opt(self.decoded_image, groundtruth)
        
            
    def encdec_handler(self, measurement, initial_net, groundtruth, sense_mat, sense_cross, keep_probability, 
                       phase_train=True, bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
        batch_norm_params = {
            # Decay for the moving averages.
            'decay': 0.995,
            # epsilon to prevent 0s in variance.
            'epsilon': 0.001,
            # force in-place updates of mean and variance estimates
            'updates_collections': None,
            'scale':True,
            'is_training':self.is_training,
            # Moving averages ends up in the trainable variables collection
            'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],}
        with slim.arg_scope([slim.conv2d, slim.fully_connected,slim.conv2d_transpose],
                            weights_initializer=slim.initializers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params):
            network_inputs = (measurement, initial_net, groundtruth, sense_cross)
            return self.encoder_decoder(network_inputs,is_training=self.is_training,dropout_keep_prob=self.keep_prob,reuse=reuse)
        
    def encoder_decoder(self, inputs, is_training=True, dropout_keep_prob=0.8, reuse=None, scope='generator'):
        (measurement, initial_net, groundtruth, sense_cross) = inputs
        self.LineAggre,self.FreqModes,self.ShrinkOpers,self.Multipliers = [],[],[],[]
        with tf.variable_scope(scope, 'generator', [inputs], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=is_training):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],stride=1, padding='SAME'):
                    self.LineAggre.append(self.LinearProj_orig(sense_cross, initial_net))
                    self.FreqModes.append(self.Domain_Transform(self.LineAggre[0], 0))
                    self.ShrinkOpers.append(self.ShrinkOper_orig(self.FreqModes[0]))
                    self.Multipliers.append(self.Multiplier_orig(self.FreqModes[0], self.ShrinkOpers[0]))
                    for stage in range(1,self.stages/2):
                        self.LineAggre.append(self.LinearProj_mid(sense_cross, initial_net, self.ShrinkOpers[-1], 
                                                              self.Multipliers[-1], stage))
                        self.FreqModes.append(self.Domain_Transform(self.LineAggre[stage], stage))
                        self.ShrinkOpers.append(self.ShrinkOper_mid(self.FreqModes[stage],self.Multipliers[-1],stage))
                        self.Multipliers.append(self.Multiplier_mid(self.FreqModes[stage],self.ShrinkOpers[stage],
                                                                    self.Multipliers[-1],stage))
                    for stage in range(self.stages/2,self.stages):
                        self.LineAggre.append(self.LinearProj_mid(sense_cross, initial_net, self.ShrinkOpers[-1], 
                                                              self.Multipliers[-1], stage))
                        self.FreqModes.append(self.Domain_Transform(self.LineAggre[stage], stage))
                        self.ShrinkOpers.append(self.ShrinkOper_mid(self.FreqModes[stage],self.Multipliers[-1],stage))
                        self.Multipliers.append(self.Multiplier_mid(self.FreqModes[stage],self.ShrinkOpers[stage],
                                                                    self.Multipliers[-1],stage))
                    output = self.LinearProj_end(initial_net, self.ShrinkOpers[-1], self.Multipliers[-1], stage+1)
                    return output
                
    def metric_opt(self, model_output, ground_truth):
        mask=None
        if self.loss_func == 'MSE':
            self.loss = loss_mse(model_output, ground_truth, mask)
        elif self.loss_func == 'RMSE':
            self.loss = loss_rmse(model_output, ground_truth, mask)
        elif self.loss_func == 'MAE':
            self.loss = loss_mae(model_output, ground_truth, mask)
        elif self.loss_func == 'SSIM':
            self.loss = loss_SSIM(model_output, ground_truth, self.mask)
        else:
            self.loss = loss_rmse(model_output, ground_truth, mask)
            
        self.metrics = calculate_metrics(model_output, ground_truth, mask)
        global_step = tf.train.get_or_create_global_step()
            
        if self.is_training:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            tvars = tf.trainable_variables()
            grads = tf.gradients(self.loss, tvars)
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step, name='train_op')
        self.info_merge = tf.summary.merge_all()
             
    def Domain_Transform(self, video, stage):
        '''
        Input Arguments:
            [batch, height, width, depth] video
            () stage: diffienciate the scope
        Return:
            [#filters][batch, height, width, depth]
        '''
        transform = []
        with tf.variable_scope('DomainTrans_%d'%(stage)):
            for ind_pattern in range(self.num_pattern):
                pattern = slim.fully_connected(video,self.ratio,scope='trans_%d'%(ind_pattern),activation_fn=None)
                transform.append(pattern)
        return transform
    
    def LinearProj_orig(self, phi_cross, phi_T_y):
        '''
        Input Argument:
            [batch, height, width, depth] phi_T_y: One step reconstruction initialization
            [height, width, depth, depth] phi_cross: phi_T_Phi the inner product of each tube
        Return:
            [batch, height, width, depth] Reconstruction Result
        '''
        gamma,rho = [],[]    
        with tf.variable_scope('LinearProj_init',None,reuse=None):
            with slim.arg_scope([slim.variable],dtype=tf.float32,initializer=slim.initializers.xavier_initializer(),
                                regularizer=slim.l2_regularizer(0.0),trainable=self.is_training):
                for ind_pattern in range(self.num_pattern):
                    rho.append(slim.variable(name='rho_%d'%(ind_pattern),shape=[]))
                    gamma.append(slim.variable(name='gamma_%d'%(ind_pattern),shape=[1,1,self.ratio,self.ratio]))
                return self.Sparse_Inverse(phi_cross, rho, gamma, phi_T_y)
    
    def ShrinkOper_orig(self, freq_mode):
        '''
        Input Argument:
            [#Patterns][batch height, width, depth]
        Return:
            [#Patterns][batch height, width, depth]
        '''
        shrinkage = []
        with tf.variable_scope('Shrinkage_init'):
            for ind_pattern in range(self.num_pattern):
                pattern = freq_mode[ind_pattern]
                pattern = slim.conv2d(pattern,self.num_kernel,5,scope='shrink_%d_0'%(ind_pattern))
                pattern = slim.conv2d(pattern,self.num_kernel,3,scope='shrink_%d_1'%(ind_pattern))
                pattern = slim.conv2d(pattern,self.ratio,3,scope='shrink_%d_2'%(ind_pattern),activation_fn=None)
                shrinkage.append(pattern)
        return shrinkage
    
    def Multiplier_orig(self, freq_mode, shrinkage):
        '''
        Input Arguments:
            freq_mode [#Patterns][batch, height, width, depth]
            shrinkage [#Patterns][batch, height, width, depth]
        Return:
            multiplier[#Patterns][batch, height, width, depth]
        '''
        eta,multiplier = [],[]
        with tf.variable_scope('multiplier_init'):
            with slim.arg_scope([slim.variable],dtype=tf.float32,initializer=slim.initializers.xavier_initializer(),
                                regularizer=slim.l2_regularizer(0.0),trainable=self.is_training):
                for ind_pattern in range(self.num_pattern):
                    eta.append(slim.variable(name='eta_%d'%(ind_pattern),shape=[]))
                    multiplier.append(tf.multiply(eta[ind_pattern], (freq_mode[ind_pattern]-shrinkage[ind_pattern])))
        return multiplier
    
    def LinearProj_mid(self, phi_cross, phi_T_y, shrinkage, multiplier, stage):
        '''
        Input Argument:
            [batch, height, width, depth] phi_T_y: One step reconstruction initialization
            [height, width, depth, depth] phi_cross: phi_T_Phi the inner product of each tube
        Return:
            [batch, height, width, depth] Reconstruction Result
        '''
        gamma,rho = [],[]    
        with tf.variable_scope('LinearProj_%d'%(stage),reuse=None):
            with slim.arg_scope([slim.variable],dtype=tf.float32,initializer=slim.initializers.xavier_initializer(),
                                regularizer=slim.l2_regularizer(0.0),trainable=self.is_training):
                pattern_aggr = phi_T_y
                for ind_pattern in range(self.num_pattern):
                    rho.append(slim.variable(name='rho_%d'%(ind_pattern),shape=[]))
                    gamma.append(slim.variable(name='gamma_%d'%(ind_pattern),shape=[1,1,self.ratio,self.ratio]))
                    auxli_v = shrinkage[ind_pattern]-multiplier[ind_pattern]
                    auxli_v = rho[-1]*slim.fully_connected(auxli_v,self.ratio,scope='LiPro_%d_1'%(ind_pattern))
                    auxli_v = slim.fully_connected(auxli_v,self.ratio,activation_fn=None,scope='LiPro_%d_End'%(ind_pattern))
                    pattern_aggr += auxli_v
                return self.Sparse_Inverse(phi_cross, rho, gamma, pattern_aggr)
    
    def ShrinkOper_mid(self, freq_mode, multiplier, stage):
        '''
        Input Argument:
            [#Patterns][batch height, width, depth]
        Return:
            [#Patterns][batch height, width, depth]
        '''
        shrinkage = []
        with tf.variable_scope('Shrinkage_%d'%(stage),reuse=None):
            for ind_pattern in range(self.num_pattern):
                pattern = freq_mode[ind_pattern]+multiplier[ind_pattern]
                pattern = slim.conv2d(pattern,self.num_kernel,5,scope='shrink_%d_0'%(ind_pattern))
                pattern = slim.conv2d(pattern,self.num_kernel,3,scope='shrink_%d_1'%(ind_pattern))
                pattern = slim.conv2d(pattern,self.ratio,3,scope='shrink_%d_2'%(ind_pattern),activation_fn=None)
                shrinkage.append(pattern)
        return shrinkage
    
    def Multiplier_mid(self, freq_mode, shrinkage, multiplier_past, stage):
        '''
        Input Arguments:
            freq_mode [#Patterns][batch, height, width, depth]
            shrinkage [#Patterns][batch, height, width, depth]
        Return:
            multiplier[#Patterns][batch, height, width, depth]
        '''
        eta,multiplier = [],[]
        with tf.variable_scope('multiplier_%d'%(stage),reuse=None):
            with slim.arg_scope([slim.variable],dtype=tf.float32,initializer=slim.initializers.xavier_initializer(),
                                regularizer=slim.l2_regularizer(0.0),trainable=self.is_training):
                for ind_pattern in range(self.num_pattern):
                    eta.append(slim.variable(name='eta_%d'%(ind_pattern),shape=[]))
                    temp = tf.multiply(eta[ind_pattern], (freq_mode[ind_pattern]-shrinkage[ind_pattern]))
                    multiplier.append(multiplier_past[ind_pattern] + temp)
        return multiplier
    
    def LinearProj_end(self, phi_T_y, shrinkage, multiplier, stage):
        '''
        Input Argument:
            [batch, height, width, depth] phi_T_y: One step reconstruction initialization
        Return:
            [batch, height, width, depth] Reconstruction Result
        '''
        gamma,rho = [],[]    
        with tf.variable_scope('LinearProj_%d'%(stage),reuse=None):
            with slim.arg_scope([slim.variable],dtype=tf.float32,initializer=slim.initializers.xavier_initializer(),
                                regularizer=slim.l2_regularizer(0.0),trainable=self.is_training):
                pattern_aggr = phi_T_y/tf.constant(self.ratio,dtype=tf.float32)
                for ind_pattern in range(self.num_pattern):
                    pattern_com = shrinkage[ind_pattern]-multiplier[ind_pattern]
                    pattern_com = slim.fully_connected(pattern_com,self.trans_dim,scope='trans_%d_0'%(ind_pattern))
                    pattern_com = slim.fully_connected(pattern_com,self.trans_dim,scope='trans_%d_1'%(ind_pattern))
                    pattern_com = slim.fully_connected(pattern_com,self.ratio,scope='trans_%d_2'%(ind_pattern),
                                                       activation_fn=None)
                    pattern_aggr += pattern_com
                for layer_cnt in range(self.num_endlayer):
                    pattern_aggr = slim.conv2d(pattern_aggr,self.num_kernel,3,scope='End_%d'%(layer_cnt))
        return slim.conv2d(pattern_aggr,self.ratio,3,scope='Final',activation_fn=tf.nn.sigmoid)
    
    def Sparse_Inverse(self, phi_cross, rho, transmtx, video):
        '''
        Input Argument:
            [height, width, depth, depth] phi_cross: the result of \phi^T \phi tensor
            [#Patterns][()] rho: the filter of different number
            [#Patterns][1,1,ratio,ratio] transmtx filter: size corresponding to the compressive ratio
            [batch, height, width, depth] video: the calculation result of the other part
        Return
            sparse tensor of (\phi^T \phi + sum \rho \D^T \D)^{-1}(video) [batch, height, width, depth]
        '''
        (height,width,ratio,ratio) = phi_cross.get_shape().as_list()        
        Sigma = phi_cross + self.noise
        for ind_pattern in range(self.num_pattern):
            Sigma += tf.multiply(rho[0],tf.matmul(transmtx[0],tf.transpose(transmtx[0],[0,1,3,2])))
        aggregator = tf.matmul(tf.matrix_inverse(Sigma),tf.transpose(video,[1,2,3,0]))
        return tf.transpose(aggregator,[3,0,1,2])
    
    def initial_parameter(self):
        config = self.config
        self.keep_prob = float(config.get('keep_rate_forward',0.8))
        # Parameter Initialization of Data Assignment
        self.stages = int(config.get('num_stages',2))
        self.num_pattern = int(config.get('num_pattern',10))
        self.batch_size = int(config.get('batch_size',12))
        self.trans_dim = int(config.get('trans_dim',16))
        self.num_kernel = int(config.get('num_kernel',16))
        self.num_endlayer = int(config.get('num_endlayer',3))
        noise_coe = float(config.get('noise',0.000001))*np.identity(self.ratio)
        self.noise = tf.expand_dims(tf.expand_dims(tf.constant(noise_coe,dtype=tf.float32),0),0)