from model import *
from cfg import hp
import matplotlib.pyplot as plt
from preprocess import *


def evaluation(saved_model_path):
    a = CreateStockSamples(hp.input_time_step, hp.output_time_step)
    x_test, y_test, scaler = a.create_samples('test')
    scale = scaler.data_range_
    feature = x_test.shape[-1]  # value:81
    # build the model ##################################################################################################
    da = DaRnn(hp.input_time_step, hp.output_time_step, feature, hp.num_layers, hp.drop_out_rate, hp.batch_size)
    input_feature = da.feature_inp  # placeholder
    label = da.label
    encoder_output, state = da.Encoder(input_feature, hp.num_hidden, hp.drop_out_rate)
    decoder_output = da.Decoder(encoder_output, state, hp.num_hidden, hp.drop_out_rate)
    decoder_output = [tf.expand_dims(i, axis=-1) for i in decoder_output]
    decoder_output = tf.concat(decoder_output, axis=1)
    # set the loss function ############################################################################################
    RMSE = tf.sqrt(tf.reduce_mean(tf.abs((decoder_output - label)) ** 2))
    MAE = tf.reduce_mean(tf.abs(decoder_output - label))
    MAPE = tf.reduce_mean(tf.abs((decoder_output - label) / label))

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        saver.restore(sess, saved_model_path)
        # new_saver = tf.train.import_meta_graph(saved_model_path + '.meta')
        # new_saver.restore(sess, saved_model_path)
        test_RSME = []
        test_MAE = []
        test_MAPE = []
        output = []
        for j in range(0, x_test.shape[0], hp.batch_size):
            end_batch = np.minimum(x_test.shape[0], j + hp.batch_size)
            batch_valid_nume = x_test[j:end_batch]
            batch_label = y_test[j:end_batch]
            _RSME, _MAE, _MAPE, decoder_result = sess.run([RMSE, MAE, MAPE, decoder_output],
                                                          feed_dict={input_feature: batch_valid_nume,
                                                                     label: batch_label})
            test_RSME.append(_RSME)
            test_MAE.append(_MAE)
            test_MAPE.append(_MAPE)
            output.append(decoder_result)
        output = np.array(output).reshape(-1, 1)
        output = scaler.inverse_transform(output)
        y_test = scaler.inverse_transform(y_test)
        test_RSME = np.mean(test_RSME * scale)
        test_MAE = np.mean(test_MAE * scale)
        test_MAPE = np.mean(test_MAPE)
        # print test and best result ###############################################################################
        print('test RMSE: {:.5f} | test MAE {:.5f} | test MAPE {:.5f}'.format(test_RSME, test_MAE, test_MAPE))
        plt.plot(output, label='predict')
        plt.plot(y_test, label='truth')
        plt.legend()
        plt.show()
    return test_RSME, test_MAE, test_MAPE


if __name__ == '__main__':
    hp.batch_size = 1
    test_RSME, test_MAE, test_MAPE = evaluation('saved_model/best_model')
