from model_mix import *
from preprocess.tools import *
import os
from cfg import *
import h5py
import tensorflow as tf


def read_samples_from_h5(data_path, station,m,ft,sample_type='train', IsGlobal=True):
    f_x_local = h5py.File(data_path+str(station)+'_'+str(m)+'_'+ft+'_x_local_{0}.h5'.format(sample_type), 'r')
    f_y = h5py.File(data_path+str(station)+'_'+str(m)+'_'+ft+'_y_{0}.h5'.format(sample_type), 'r')
    x_local = []
    for i in range(len(f_x_local.keys())):
        key = 'x_local_{0}'.format(i)
        x_local.append(f_x_local[key][:])    
    y_sample = f_y['y_{0}'.format(sample_type)][:]
    if IsGlobal:
        x_global = []
        f_x_global = h5py.File(data_path+str(station)+'_'+str(m)+'_'+ft+'_x_global_{0}.h5'.format(sample_type), 'r')
        for j in range(len(f_x_global.keys())):
            key = 'x_global_{0}'.format(j)
            x_global.append(f_x_global[key][:])
        f_x_global.close()
    f_x_local.close()    
    f_y.close()
    if IsGlobal:
        return [x_local, x_global], y_sample
    else:
        return x_local, y_sample    

######################################################################################
######################################################################################
######################## do configuration and set hyperparameters ####################
######################################################################################
######################################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



from numpy.random import seed
seed(111)
from tensorflow import set_random_seed
set_random_seed(111)
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)


m = 72
ft = 'O3'
data_path = 'data/train/'
x_test, y_test = read_samples_from_h5(data_path, ph.name, m, ft, sample_type='test', IsGlobal=False)
x_train, y_train = read_samples_from_h5(data_path, ph.name, m, ft, sample_type='train', IsGlobal=False)


x_train_encoder_nume, x_train_encoder_category, \
x_train_decoder_nume, x_train_decoder_category, y_train_init = data_transform(x_train)

if ft == 'PM2.5':
    x_train_decoder_nume = x_train_decoder_nume[:,:,2:]
embeding_para_encoder = np.array([25, 17, 24, 7])
embeding_para_decoder = np.array([25, 24, 7])
x_test_encoder_nume, x_test_encoder_category, \
x_test_decoder_nume, x_test_decoder_category, y_test_init = data_transform(x_test)
if ft == 'PM2.5':
    x_test_decoder_nume = x_test_decoder_nume[:,:,2:]
inp_nume = x_train_encoder_nume.shape[-1]
inp_categ = x_train_encoder_category.shape[-1]
external_nume = x_train_decoder_nume.shape[-1]
external_category = x_train_decoder_category.shape[-1]

#######################################################################################
#######################################################################################
######################### build the model #############################################
#######################################################################################
#######################################################################################

da_lstm = DA_RNN(hp.time_step, hp.output_time_step, inp_nume, inp_categ,external_nume,
                 external_category,
                 hp.num_layers, hp.drop_out_rate, hp.batch_size, hp.embedding_dim,
                 embeding_para_encoder, embeding_para_decoder)
nume_feature = da_lstm.nume_inp
categ_feature = da_lstm.categ_inp
external_nume = da_lstm.external_decoder_nume
external_category = da_lstm.external_decoder_category
y_init = da_lstm.y_initial
label = da_lstm.label
# def Encoder(self, nume_feature, categ_feature, num_hidden, drop_out_rate )
encoder_output, state = da_lstm.Encoder(nume_feature, categ_feature, hp.num_hidden, hp.drop_out_rate)


decoder_output = da_lstm.Decoder(encoder_output, state, external_nume, external_category, y_init,
                                 hp.num_hidden, hp.drop_out_rate)
print('decoder_output.shape1:',decoder_output[0].shape)
decoder_output = [tf.expand_dims(i, axis=-2) for i in decoder_output]
print('decoder_output.shape2:',decoder_output[0].shape)
decoder_output = tf.concat(decoder_output, axis=1)
print('decoder_output.shape3:',decoder_output.shape)

#######################################################################################
#######################################################################################
################################# build the loss and train operation ##################
#######################################################################################
#######################################################################################
smape = 2*tf.reduce_mean(tf.abs(decoder_output - label + 1e-8)/(tf.abs(decoder_output) + tf.abs(label) + 1e-5))
mae = tf.reduce_mean(tf.abs(decoder_output - label))
mse = tf.reduce_mean(tf.square(decoder_output - label))
loss = mae + smape
train_op = tf.train.AdamOptimizer(0.001)
grads_and_vars = train_op.compute_gradients(loss)
gradients, variables = zip(*grads_and_vars)
clipped_gradients, glob_norm = tf.clip_by_global_norm(gradients, hp.max_gradients)
train_op = train_op.apply_gradients(zip(clipped_gradients, variables))
#######################################################################################
#######################################################################################
################################ begin training #######################################
#######################################################################################
#######################################################################################
sess.run(tf.global_variables_initializer())
best_smape = []
best_mae = []
best_mse = []
best_smape_p = np.inf
saver = tf.train.Saver()
if not os.path.exists('saved_model'):
    os.mkdir('saved_model')
for i in range(hp.epoch):
    # training ##############################
    train_smape = []
    train_mae = []
    train_mse = []
    for j in range(0, x_train_encoder_nume.shape[0], hp.batch_size):
        end_batch = np.minimum(x_train_encoder_nume.shape[0], j + hp.batch_size)
        batch_train_nume = x_train_encoder_nume[j:end_batch,:,:]
        batch_train_categ = x_train_encoder_category[j:end_batch,:,:]
        batch_label = y_train[j:end_batch,:,:]
        batch_train_external_nume = x_train_decoder_nume[j:end_batch,:,:]
        batch_train_external_category = x_train_decoder_category[j:end_batch,:,:]
        batch_train_y_init = y_train_init[j:end_batch,:,:]
        _, _loss, _smape, _mae, _mse = sess.run([train_op, loss, smape, mae, mse],
                                        feed_dict={nume_feature: batch_train_nume,
                                                   categ_feature: batch_train_categ,
                                                   external_nume: batch_train_external_nume,
                                                   external_category: batch_train_external_category,
                                                   y_init: batch_train_y_init,
                                                   label: batch_label})

        if j%(hp.batch_size*10) == 0:
            print('epoch: {} | total loss: {:.5f} | smape: {:.5f} | mae: {:.5f} | mse: {:.5f} '
                  .format(i, _loss, _smape, _mae, _mse))
        train_mae.append(_mae)
        train_mse.append(_mse)
        train_smape.append(_smape)
    train_mae = np.mean(train_mae)
    train_mse = np.mean(train_mse)
    train_smape = np.mean(train_smape)
    print('epoch: {} | train smape: {:.5f} | train mae: {:.5f} | train mse: {:.5f} '
          .format(i, _loss, train_smape, train_mae, train_mse))

    #  validating #############################
    val_smape = []
    val_mae = []
    val_mse = []
    k = 0
    for j in range(0, x_test_encoder_nume.shape[0], hp.batch_size):
        k += 1
        end_batch = np.minimum(x_test_encoder_nume.shape[0], j + hp.batch_size)
        batch_test_nume = x_test_encoder_nume[j:end_batch, :, :]
        batch_test_categ = x_test_encoder_category[j:end_batch, :, :]
        batch_label = y_test[j:end_batch, :, :]
        batch_test_external_nume = x_test_decoder_nume[j:end_batch, :, :]
        batch_test_external_category = x_test_decoder_category[j:end_batch, :, :]
        batch_test_y_init = y_test_init[j:end_batch,:,:]
        _loss, _smape, _mae, _mse = sess.run([loss, smape, mae, mse],
                                                feed_dict={nume_feature: batch_test_nume,
                                                           categ_feature: batch_test_categ,
                                                           external_nume: batch_test_external_nume,
                                                           external_category: batch_test_external_category,
                                                           y_init: batch_test_y_init,
                                                           label: batch_label})

        val_mae.append(_mae)
        val_mse.append(_mse)
        val_smape.append(_smape)
    val_mae = np.mean(val_mae)
    val_mse = np.mean(val_mse)
    val_smape = np.mean(val_smape)
    best_mae.append(val_mae)
    best_mae2 = np.min(best_mae)
    best_mse.append(val_mse)
    best_mse2 = np.min(best_mse)
    best_smape.append(val_smape)
    best_smape2 = np.min(best_smape)
    print('validate smape: {:.5f} | validate mae {:.5f} | validate mse {:.5f}'.format(val_smape, val_mae, val_mse))
    print('best smape: {:.5f} | best mae {:.5f} | best mse {:.5f}'.format(best_smape2, best_mae2, best_mse2))
    if best_smape_p>best_smape2:
        best_smape_p = best_smape2
        print('saving current best model!!!')
        saver.save(sess, 'saved_model/best_model_mix_{0}'.format(ft), global_step = i)
sess.close()
