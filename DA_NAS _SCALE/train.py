from model import *
from preprocess import *
import os
import numpy as np
from cfg import hp

# do configuration #####################################################################################################
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)
# load and process the data ###########################################################################################
stock = CreateStockSamples(hp.input_time_step, hp.output_time_step)
x_train, y_train, _ = stock.create_samples('train')
x_valid, y_valid, _ = stock.create_samples('valid')
# x_test, y_test = a.create_samples('test')
feature = x_train.shape[-1]            # value:81
# build the model ######################################################################################################
da = DaRnn(hp.input_time_step, hp.output_time_step, feature, hp.num_layers, hp.drop_out_rate, hp.batch_size)
input_feature = da.feature_inp    # placeholder
label = da.label
encoder_output, state = da.Encoder(input_feature, hp.num_hidden, hp.drop_out_rate)
decoder_output = da.Decoder(encoder_output, state, hp.num_hidden, hp.drop_out_rate)
decoder_output = [tf.expand_dims(i, axis=-1) for i in decoder_output]
decoder_output = tf.concat(decoder_output, axis=1)
# set the loss function ################################################################################################
RMSE = tf.sqrt(tf.reduce_mean(tf.abs((decoder_output - label))**2))
MAE = tf.reduce_mean(tf.abs(decoder_output - label))
MAPE = tf.reduce_mean(tf.abs((decoder_output - label)/label))
loss = RMSE + MAE
# set optimizer ########################################################################################################
train_op = tf.train.AdamOptimizer(0.001)
grads_and_vars = train_op.compute_gradients(loss)
gradients, variables = zip(*grads_and_vars)
clipped_gradients, glob_norm = tf.clip_by_global_norm(gradients, hp.max_gradients)
train_op = train_op.apply_gradients(zip(clipped_gradients, variables))
# begin training #######################################################################################################
sess.run(tf.global_variables_initializer())
best_RSME = np.inf
best_MAE = np.inf
best_MAPE = np.inf
best_loss = np.inf
# recurrent ############################################################################################################
for i in range(1, hp.epoch):
    for j in range(0, x_train.shape[0], hp.batch_size):
        end_batch = np.minimum(x_train.shape[0], j + hp.batch_size)
        batch_train = x_train[j:end_batch]
        batch_label = y_train[j:end_batch]
        _, _loss, _RSME, _MAE, _MAPE = sess.run([train_op, loss, RMSE, MAE, MAPE],
                                                feed_dict={input_feature: batch_train, label: batch_label})
        if j % (hp.batch_size * 20) == 0:
            print('Epoch: {} | Total Loss: {:.5f} | RMSE: {:.5f} | MAE: {:.5f} | MAPE: {:.5f} '
                  .format(i, _loss, _RSME, _MAE, _MAPE))
    # validating #######################################################################################################
    valid_RSME = []
    valid_MAE = []
    valid_MAPE = []
    k = 0
    for j in range(0, x_valid.shape[0], hp.batch_size):
        k += 1
        end_batch = np.minimum(x_valid.shape[0], j + hp.batch_size)
        batch_valid_nume = x_valid[j:end_batch]
        batch_label = y_valid[j:end_batch]
        _RSME, _MAE, _MAPE = sess.run([RMSE, MAE, MAPE],
                                      feed_dict={input_feature: batch_valid_nume, label: batch_label})
        valid_RSME.append(_RSME)
        valid_MAE.append(_MAE)
        valid_MAPE.append(_MAPE)
    valid_RSME = np.mean(valid_RSME)
    valid_MAE = np.mean(valid_MAE)
    valid_MAPE = np.mean(valid_MAPE)
    valid_loss = valid_RSME + valid_MAE
    # print validate and best result ###################################################################################
    print('      valid RMSE: {:.5f} | valid MAE {:.5f} | valid MAPE {:.5f}'.format(valid_RSME, valid_MAE, valid_MAPE))
    print('      best  RMSE: {:.5f} | best  MAE {:.5f} | best  MAPE {:.5f}'.format(best_RSME, best_MAE, best_MAPE))
    # save the best model ##############################################################################################
    if best_loss > valid_loss:
        best_RSME = valid_RSME
        best_MAE = valid_MAE
        best_MAPE = valid_MAPE
        best_loss = valid_loss
        print('..........................waiting for saving current best model..........................')
        writer = tf.summary.FileWriter("D://TensorBoard//test", sess.graph)
        saver = tf.train.Saver()
        if not os.path.exists('saved_model'):
            os.mkdir('saved_model')
        saver.save(sess, 'saved_model/best_model')
sess.close()
