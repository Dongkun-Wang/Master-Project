import tensorflow as tf
import numpy as np
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.contrib import rnn


class DA_RNN(object):
    def __init__(self, time_step, output_time_step, inp_nume, inp_categ, external_nume,
                 external_category,
                 num_layers, drop_out_rate, batch_size, embedding_dim, embeding_para_encoder,
                 embeding_para_decoder):
        self.time_step = time_step
        self.inp_nume = inp_nume
        self.inp_categ = inp_categ
        self.inp_features = inp_nume + inp_categ*embedding_dim
        self.num_layers = num_layers
        self.drop_out_rate = drop_out_rate
        self.batch_size = batch_size
        self.output_time_step = output_time_step
        self.nume_inp = tf.placeholder(shape = [None, self.time_step, self.inp_nume], dtype = tf.float32, name='numerical')
        self.categ_inp = tf.placeholder(shape = [None, self.time_step, self.inp_categ], dtype = tf.int64, name='categorical')
        self.y_initial = tf.placeholder(shape = [None, self.time_step, 1], dtype= tf.float32, name='y_init')
        self.label = tf.placeholder(shape = [None, self.time_step, 1], dtype = tf.float32, name='label')
        self.embeding_para_encoder = embeding_para_encoder
        self.embeding_para_decoder = embeding_para_decoder
        self.external_nume = external_nume
        self.external_category = external_category
        self.embeding_dim = embedding_dim
        self.external_decoder_nume = tf.placeholder(shape = [None, self.time_step, self.external_nume], dtype = tf.float32)
        self.external_decoder_category = tf.placeholder(shape = [None, self.time_step, self.external_category], dtype = tf.int64)

    def embedding(self, inputs, feature_encode, embedded_size, zero_pad=False, scale=True,
                  scope='embedding', reuse=None):
        """Embeds a given tensor

        Args:
            inputs: Tensor with type int32 or int 64 containing the ids of the features
            feature_encode: an int, feature coding size
            embedded_size: an int. number of embedding hidden units
            zero_pad: boolean, True all the values of first row should be constant zeros
            scale: boolean, if True the outputs is multipilzed by sqrt embedded_size
            scope: Optional scope for variable_scope
            reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name

        returns:
            a tensor with one more rank than inputs's, last dimension is embedded_size
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            lookup_table = tf.get_variable('lookup_table', dtype=tf.float32,
                                           shape=[feature_encode, embedded_size],
                                           initializer=tf.random_normal_initializer(0., embedded_size ** -0.5))
            lookup_table = tf.nn.sigmoid(lookup_table)
            outputs = tf.nn.embedding_lookup(lookup_table, inputs)

            if scale:
                outputs = outputs * (embedded_size ** 0.5)

        return outputs

    def spatial_attention(self,
                      input,
                      attention_input,
                      last_state,
                      virtual_state,
                      scope):
            """
            Compute spatial attention given inputs and attention inputs
            :param input: tensor, shape:[batch_size, 1, num_features]
            :param attention_input: shape:[batch_size, time_steps, num_features]
            :param last_state:  shape: list, contains the states of the number of layers.
            :return: A tensor multiplied by the attention weights, shape: [batch_size, 1, num_features]
            """
            with tf.variable_scope('spatial_attention' ):

                last_state = nest.flatten(last_state)
                last_state = tf.concat(last_state, axis=1)
                n = last_state.get_shape()[-1].value
                if virtual_state is not None:
                    # virtual_state = nest.flatten(virtual_state)
                    # virtual_state = tf.concat(virtual_state, axis = -1)
                    last_state = tf.concat([last_state, virtual_state], axis = -1)
                    virtual_transform = tf.get_variable(name='virtual_transform',
                                                       shape=[last_state.get_shape()[-1],
                                                       n])
                    last_state = tf.matmul(last_state, virtual_transform)
                last_state = tf.expand_dims(last_state, axis = -1)
                last_state = tf.concat([last_state for _ in range(self.inp_features)], axis = 2)
                dim = last_state.get_shape()[1].value
                U_e = tf.get_variable(shape = [1, self.time_step, 1, self.time_step], name = 'U_e')
                W_e = tf.get_variable(shape=[1, dim, 1, self.time_step], name='W_e')
                V_e = tf.get_variable(shape=[1, 1, self.time_step, 1], name='V_e')

                U_e_x_k = self.hd_matmul(U_e, attention_input)
                W_e_h_s = self.hd_matmul(W_e, last_state)
                attention_tanh = tf.nn.tanh(U_e_x_k + W_e_h_s)
                attention_tanh = tf.transpose(attention_tanh, [0,1,3,2])
                alpha_t = tf.nn.conv2d(attention_tanh, V_e, strides = [1,1,1,1], padding = 'VALID')
                alpha_t = tf.reshape(alpha_t, [-1,1,self.inp_features])
                alpha_t = tf.nn.sigmoid(alpha_t)
                output = input*alpha_t
                return output


    def Encoder(self, nume_feature, categ_feature, num_hidden, drop_out_rate ):
        embeding_categ_feature = [self.embedding(categ_feature[:,:,i], self.embeding_para_encoder[i],self.embeding_dim, scope=str(i))
                                  for i in range(self.embeding_para_encoder.shape[0])]
        embeding_categ_feature = tf.concat(embeding_categ_feature, axis = -1)
        encoder_inputs2 = tf.concat([nume_feature, embeding_categ_feature], axis = -1)
        attention_input = encoder_inputs2
        encoder_inputs = tf.split(encoder_inputs2, self.time_step, 1)
        with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
            cells = []
            for i in range(self.num_layers):
                cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, state_is_tuple= True)
                if drop_out_rate:
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                       output_keep_prob= 1 - drop_out_rate,
                    )
                cells.append(cell)
            encoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            batch_size = array_ops.shape(encoder_inputs[0])[0]
            state = encoder_cell.zero_state(batch_size, dtype = tf.float32)
            encoder_output = []
            for i, input_per_time in enumerate(encoder_inputs):
                if i>0:
                    tf.get_variable_scope().reuse_variables()
            # ######################## virtual attention #####################################
                encoder_attention_input = self.spatial_attention(input_per_time,
                                                        attention_input,
                                                        state,
                                                        None,
                                                        '1')
                encoder_attention_input = tf.reshape(encoder_attention_input, [-1, self.inp_features])
                # encoder_attention_input = tf.reshape(input_per_time, [-1, self.inp_features])
                output, state_virtual = encoder_cell(encoder_attention_input, state)

            # ################################################################################
            ##     encoder_attention_input = tf.reshape(encoder_attention_input, [-1, self.inp_features])
                encoder_attention_input = self.spatial_attention(input_per_time,
                                                        attention_input,
                                                        state_virtual,
                                                        None,
                                                       # state_virtual,
                                                        '1')
                encoder_attention_input = tf.reshape(encoder_attention_input, [-1, self.inp_features])
                output, state = encoder_cell(encoder_attention_input, state)
            #################################################################################

                output = tf.reshape(output, [-1,1, output.get_shape()[-1].value])
                encoder_output.append(output)
        return encoder_output, state

    def temporal_attention(self,
                          input,
                          last_state,
                           virtual_state,
                           num_hidden):
        """
        Compute spatial attention given inputs and attention inputs
        :param input: tensor, shape:[batch_size, 1, num_features]
        :param attention_input: shape:[batch_size, time_steps, num_features]
        :param last_state:  shape: list, contains the states of the number of layers.
        :return: A tensor multiplied by the attention weights, shape: [batch_size, 1, num_features]
        """
        with tf.variable_scope('temporal_attention'):
            last_state = nest.flatten(last_state)
            last_state = tf.concat(last_state, axis = 1)
            if virtual_state is not None:
                virtual_transform = tf.get_variable(name = 'virtual_transform_decoder', shape = [last_state.get_shape()[-1].value*2, last_state.get_shape()[-1].value])
                virtual_state = nest.flatten(virtual_state)
                virtual_state = tf.concat(virtual_state, axis = 1)
                last_state = tf.concat([virtual_state, last_state], axis = -1)
                last_state = tf.matmul(last_state, virtual_transform)
            last_state = tf.expand_dims(last_state, axis = -1)
            last_state = tf.concat([last_state for _ in range(self.time_step)], axis = 2)
            dim = last_state.get_shape()[1].value
            U_d = tf.get_variable(shape = [1, num_hidden, 1, num_hidden], name = 'U_d')
            W_d = tf.get_variable(shape=[1, dim, 1, num_hidden], name='W_d')
            V_d = tf.get_variable(shape=[1, num_hidden, 1, 1], name='V_d')
            attention_input = tf.concat(input,axis = 1)
            attention_input = tf.transpose(attention_input, [0,2,1])
            U_d_h_i = self.hd_matmul(U_d, attention_input)
            W_d_h_s = self.hd_matmul(W_d, last_state)
            attention_tanh = tf.nn.tanh(W_d_h_s + U_d_h_i)
            attention_tanh = tf.transpose(attention_tanh, [0,3,2,1])
            alpha_t = tf.nn.conv2d(attention_tanh, V_d, strides = [1,1,1,1], padding = 'VALID')
            alpha_t = tf.reshape(alpha_t, [-1,1,self.time_step])
            alpha_t = tf.nn.softmax(alpha_t)
            output = attention_input*alpha_t
            output = tf.reduce_sum(output, axis = 2)
            return output

    def Decoder(self, decoder_inputs, state, external_nume, external_category, y_initial, num_hidden, drop_out_rate):
        embeding_categ_feature = [self.embedding(external_category[:,:,i], self.embeding_para_decoder[i],self.embeding_dim, scope='decoder'+str(i))
                                  for i in range(self.embeding_para_decoder.shape[0])]
        embeding_categ_feature = tf.concat(embeding_categ_feature, axis = -1)
        decoder_external = tf.concat([external_nume, embeding_categ_feature], axis = -1)
        decoder_external = tf.split(decoder_external, self.time_step, 1)
        y_initial = tf.split(y_initial, self.time_step, 1)
        with tf.variable_scope('Decoder', reuse = tf.AUTO_REUSE):
            cells = []
            for i in range(self.num_layers):
                cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, state_is_tuple= True)
                if drop_out_rate:
                    cell = tf.nn.rnn_cell.DropoutWrapper(
                        cell, output_keep_prob= 1-drop_out_rate,

                    )
                cells.append(cell)
            decoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            batch_size = array_ops.shape(decoder_inputs[0])[0]
            decoder_output = []
            for i in range(self.output_time_step):
                if i == 0:
                    y_t_1 = tf.zeros([batch_size, 1], dtype = tf.float32)
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                ####################### virtual attention ######################
                c_t = self.temporal_attention(decoder_inputs, state, None, num_hidden)
                decoder_external_tmp = tf.squeeze(decoder_external[i], axis=1)
                y_initial_tmp = tf.squeeze(y_initial[i], axis=1)
                decoder_attention_input = tf.concat([y_t_1,y_initial_tmp,c_t],axis = 1 )
                W_tilta = tf.get_variable(name = 'W_tilta', shape= [decoder_attention_input.shape[-1].value, 1], dtype = tf.float32)
                b = tf.get_variable(name = 'b', shape = 1, dtype = tf.float32)
                #
                y_t_1_tilta = tf.matmul(decoder_attention_input, W_tilta) + b
                state_tmp = []
                for i2 in state:
                    state_tmp.append(i2[1])
                state_tmp = tf.concat(state_tmp,axis = 1)

                # print(decoder_external_tmp.shape)
                decoder_input = tf.concat([state_tmp, y_t_1_tilta, decoder_external_tmp], axis = 1)
                # decoder_input = tf.concat([y_t_1, decoder_external_tmp], axis = 1)
                d_t, state_virtual = decoder_cell(decoder_input, state)
                #################################################################
                c_t = self.temporal_attention(decoder_inputs, state_virtual, None, num_hidden)
                decoder_external_tmp = tf.squeeze(decoder_external[i], axis=1)
                y_initial_tmp = tf.squeeze(y_initial[i], axis=1)
                decoder_attention_input = tf.concat([y_t_1,y_initial_tmp,c_t],axis = 1 )
                # decoder_attention_input = tf.concat([decoder_attention_input], axis = 1)
                W_tilta = tf.get_variable(name = 'W_tilta', shape= [decoder_attention_input.shape[-1].value, 1], dtype = tf.float32)
                b = tf.get_variable(name='b', shape = 1, dtype = tf.float32)
                y_t_1_tilta = tf.matmul(decoder_attention_input, W_tilta) + b
                state_tmp = []
                for i2 in state:
                    state_tmp.append(i2[1])
                state_tmp = tf.concat(state_tmp,axis = 1)

                # print(decoder_external_tmp.shape)
                decoder_input = tf.concat([state_tmp, y_t_1_tilta, decoder_external_tmp], axis = 1)
                d_t, state = decoder_cell(decoder_input, state)
                ###################################################################
                p = d_t.get_shape()[1].value
                d_t_c_t = tf.concat([d_t, c_t], axis = 1)
                # d_t_c_t = tf.concat(d_t, axis=1)
                b_w = tf.get_variable(name = 'b_w', shape = p, dtype = tf.float32)
                b_v = tf.get_variable(name = 'b_v', shape = 1, dtype = tf.float32)
                # m = num_hidden
                p_m = d_t_c_t.get_shape()[-1].value
                W_y = tf.get_variable(name = 'W_y', shape = [p_m, p])
                v_y = tf.get_variable(name = 'V_y', shape = [p, 1])
                y_t_1_tilta = tf.matmul((tf.matmul(d_t_c_t, W_y) + b_w),v_y) + b_v
                y_t_1 = y_t_1_tilta
                decoder_output.append(y_t_1_tilta)
            return decoder_output


    def hd_matmul(self, variable, input):
        attention_input = tf.expand_dims(input, axis=-1)
        attention_input = tf.transpose(attention_input, [0, 2, 1, 3])
        output = tf.nn.conv2d(attention_input, variable, strides=[1, 1, 1, 1], padding='VALID')
        output = tf.transpose(output, [0, 2, 3, 1])
        return output


if __name__ == '__main__':
    # DA_RNN(time_step, output_time_step, inp_features, num_layers, drop_out_rate, batch_size)
    da_lstm = DA_RNN(7, 16, 8,  2, 0.3, 32)
    encoder_inputs2 = tf.placeholder(shape = [None, 7, 8], dtype = tf.float32)
    # def Encoder(encoder_inputs, num_hidden, drop_out_rate)
    encoder_output, state = da_lstm.Encoder(encoder_inputs2, 128,  0.3)
    # def Decoder(self, decoder_inputs, state, num_hidden, drop_out_rate):
    decoder_output = da_lstm.Decoder( encoder_output, state, 128, 0.3)
    decoder_output = [tf.expand_dims(i,axis=-2) for i in decoder_output]
    decoder_output = tf.concat(decoder_output, axis = 1)
