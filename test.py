from __future__ import absolute_import

import tensorflow as tf
import os
import yaml
import h5py

from Model.Decoder_Handler import Decoder_Handler

config_filename = './Model/Config.yaml'
scenario = 'NBA'

def main():

    if scenario == 'NBA':
        folder_id, config_id = 'NBA-Decoder-TNN', 'config-nba.yaml'
    elif scenario == 'Crash':
        folder_id, config_id = 'Crash-Decoder-TNN-T0808105703-K0.80L0.020-RMSE', 'config_13.yaml'
    elif scenario == 'Aerial':
        folder_id, config_id = 'Aerial-Decoder-TNN-T0808105703-K0.80L0.020-RMSE', 'config_34.yaml'
    else:
        folder_id, config_id = 'NBA-Decoder-TNN-T0808105703-K0.80L0.020-RMSE', 'config_34.yaml'

    with open(config_filename) as handle:
        model_config = yaml.load(handle)
    log_dir = os.path.join(os.path.abspath('.'), model_config['result_dir'], model_config['result_model'], folder_id)

    with open(os.path.join(log_dir, config_id)) as handle:
        model_config = yaml.load(handle)
    data_name = os.path.join(os.path.abspath('.'), 'Data', model_config['category'], model_config['data_name'])
    if model_config['mask_name'] == 'Original':
        mask_name = None
    else:
        mask_name = os.path.join(os.path.abspath('.'),'Data',model_config['category'],model_config['mask_name'])
        
    dataset_name = (data_name,mask_name)
    
    tf_config = tf.ConfigProto()
    os.environ["CUDA_VISIBLE_DEVICES"] = "2" # Please change the id of GPU in your local server accordingly
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        Cube_Decoder = Decoder_Handler(dataset_name=dataset_name, model_config=model_config, sess = sess, is_training=False)
        Cube_Decoder.test()

if __name__ == '__main__':
    main()
