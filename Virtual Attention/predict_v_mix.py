from model_mix import *
from cfg import *
from preprocess.tools import *
# from model import *
# from pre_process.utils import *
import os
import glob
import numpy as np
from preprocess.predict_preprocess_v_sim import create_anhui_predict_samples
import pandas as pd


def reversed_preds(scaled_ft, scaler):
    # scaled_ft = tf.transpose(scaled_ft)
    min_ = scaler.min_
    scale_ = scaler.scale_
    min_tensor = tf.convert_to_tensor(min_, dtype=tf.float32)
    scale_tensor = tf.convert_to_tensor(scale_, dtype=tf.float32)
    return tf.divide(tf.subtract(scaled_ft, min_tensor), scale_tensor)


def eval(ft, hp, m, n, date, saved_model_path, scaler_path):
    n = 72
    a = create_anhui_predict_samples(date,ft,n)
    x_test = a()
    scaler = read_scaler(ft, num=3, scaler_path=scaler_path)
    # scale = scaler.data_range_
    embeding_para_encoder = np.array([25, 17, 24, 7])
    embeding_para_decoder = np.array([25, 24, 7])
    x_test_encoder_nume, x_test_encoder_category, \
    x_test_decoder_nume, x_test_decoder_category, y_test_init = data_transform(x_test)
    # if ft == 'PM2.5':
    #     print("executed!!!")
    #     x_test_decoder_nume = x_test_decoder_nume[:,:,2:]
    inp_nume = x_test_encoder_nume.shape[-1]
    inp_categ = x_test_encoder_category.shape[-1]
    external_nume = x_test_decoder_nume.shape[-1]
    external_category = x_test_decoder_category.shape[-1]

    da_lstm = DA_RNN(hp.time_step, hp.output_time_step, inp_nume, inp_categ, external_nume,
                     external_category,
                     hp.num_layers, hp.drop_out_rate, hp.batch_size, hp.embedding_dim,
                     embeding_para_encoder, embeding_para_decoder)
    nume_feature = da_lstm.nume_inp
    categ_feature = da_lstm.categ_inp
    external_nume = da_lstm.external_decoder_nume
    external_category = da_lstm.external_decoder_category
    y_init = da_lstm.y_initial
    # def Encoder(self, nume_feature, categ_feature, num_hidden, drop_out_rate )
    encoder_output, state = da_lstm.Encoder(nume_feature, categ_feature, hp.num_hidden, hp.drop_out_rate)

    # Decoder(decoder_inputs, state, external_nume, external_category, num_hidden, drop_out_rate):

    decoder_output = da_lstm.Decoder(encoder_output, state, external_nume, external_category, y_init, 
                                     hp.num_hidden, hp.drop_out_rate)
    decoder_output = [tf.expand_dims(i, axis=-2) for i in decoder_output]
    decoder_output = tf.concat(decoder_output, axis=1)
    decoder_output = tf.squeeze(decoder_output, axis=-1)
    if n == 3:
        initializer = tf.contrib.keras.initializers.he_normal()
        dec_weight = tf.get_variable(name="dec_out_weight", shape=[72, 3], initializer=initializer, trainable=True)
        decoder_output = tf.matmul(decoder_output, dec_weight)
    decoder_output_reverse = reversed_preds(tf.reshape(decoder_output,[-1,n]), scaler)
    saver = tf.train.Saver()
    config= tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config = config) as sess:
        try:
            model_name = 'best_model_mix_{0}'.format(ft)
            model_all = glob.glob(os.path.join(saved_model_path, model_name+'*'))
            num_list = []
            for name in model_all:
                num = int(name.split('-')[1].split('.')[0])
                num_list.append(num)
            latest = sorted(num_list, reverse=True)[0]
            model = saved_model_path+model_name+'-'+str(latest)
            print(model)
            saver.restore(sess, model)
            predicted = sess.run([decoder_output_reverse],
                                feed_dict={nume_feature: x_test_encoder_nume, categ_feature: x_test_encoder_category,
                                       external_nume: x_test_decoder_nume, external_category: x_test_decoder_category,
                                       y_init: y_test_init})
        except:
            raise ValueError("No such model trained for {0}".format(ft))
    return predicted

def virtual_RNN_predictions(date, saved_model_path, scaler_path, saved_data_path,m=72,n=3,
                           fts=['PM2.5', 'PM10', 'O3', 'SO2', 'NO2', 'CO' ],
                           saved_pred_name_start='RNN_attn_virtual_', saved_pred_name_end='_pred.pkl',hp=hp
                           ):
    predictions = []
    for ft in fts[2:3]:
        print(ft)
        tf.reset_default_graph()
        prediction = eval(ft, hp, m, n, date, saved_model_path, scaler_path)
        predictions.append(prediction[0].reshape(-1,1))
#        file_name = saved_data_path + saved_pred_name_start +str(n)+'_'+ft+saved_pred_name_end
#        save_pkl(file_name, predictions)
    return predictions


def get_embedding_lookup_table(saved_model_path, ft):
    tf.reset_default_graph()
    with tf.Session() as sess:
        try:
            model_name = 'best_model_mix_model_{0}'.format(ft)
            model_all = glob.glob(os.path.join(saved_model_path, model_name + '*'))
            num_list = []
            for name in model_all:
                num = int(name.split('-')[1].split('.')[0])
                num_list.append(num)
            latest = sorted(num_list, reverse=True)[0]
            model = saved_model_path + model_name + '-' + str(latest)
            meta_graph = model + '.meta'
            saver = tf.train.import_meta_graph(meta_graph)
            saver.restore(sess, model)
            sess.run(tf.global_variables_initializer())
            all_vars = tf.trainable_variables()
            for v in all_vars:
                print("%s with value %s"%(v.name, sess.run(v)))
        except:
            raise ValueError("No such model trained for {0}".format(ft))
    return


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    hp.batch_size = 1
    fts = ['PM2.5', 'PM10', 'O3', 'SO2', 'NO2', 'CO']
    date = ph.end_date  # the day before the local data
    saved_model_path = 'saved_model/'
    scaler_path = 'data/scaler/'
    saved_data_path = 'data/saved_data/'
    predictions = virtual_RNN_predictions(date, saved_model_path, scaler_path, saved_data_path)
    predictions = np.concatenate(predictions, axis=1)
    pred_df = pd.DataFrame(predictions, columns=['O3'])

