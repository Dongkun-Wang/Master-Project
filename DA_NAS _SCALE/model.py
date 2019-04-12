import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops


class DaRnn(object):
    def __init__(self, input_time_step, output_time_step, inp_features, num_layers, drop_out_rate, batch_size):
        self.input_time_step = input_time_step
        self.output_time_step = output_time_step
        self.inp_features = inp_features
        self.num_layers = num_layers
        self.drop_out_rate = drop_out_rate
        self.batch_size = batch_size
        self.label = tf.placeholder(shape=[None, self.output_time_step], dtype=tf.float32)
        self.feature_inp = tf.placeholder(shape=[None, self.input_time_step, self.inp_features], dtype=tf.float32)

    def hd_matmul(self, variable, inputs):
        attention_input = tf.expand_dims(inputs, axis=-1)
        attention_input = tf.transpose(attention_input, [0, 2, 1, 3])
        output = tf.nn.conv2d(attention_input, variable, strides=[1, 1, 1, 1], padding='VALID')
        output = tf.transpose(output, [0, 2, 3, 1])
        return output

    def spatial_attention(self, inputs, attention_input, last_state):
            """
            Compute spatial attention given inputs and attention inputs
            :param inputs: tensor, shape:[batch_size, 1, num_features]
            :param attention_input: shape:[batch_size, time_steps, num_features]
            :param last_state:  shape: list, contains the states of the number of layers.
            :return: A tensor multiplied by the attention weights, shape: [batch_size, 1, num_features]
            """
            with tf.variable_scope('spatial_attention'):
                last_state = nest.flatten(last_state)
                last_state = tf.concat(last_state, axis=1)
                last_state = tf.expand_dims(last_state, axis=-1)
                last_state = tf.concat([last_state for _ in range(self.inp_features)], axis=2)
                dim = last_state.get_shape()[1].value
                U_e = tf.get_variable(shape=[1, self.input_time_step, 1, self.input_time_step], name='U_e')
                W_e = tf.get_variable(shape=[1, dim, 1, self.input_time_step], name='W_e')
                V_e = tf.get_variable(shape=[1, 1, self.input_time_step, 1], name='V_e')
                U_e_x_k = self.hd_matmul(U_e, attention_input)
                W_e_h_s = self.hd_matmul(W_e, last_state)
                attention_tanh = tf.nn.tanh(U_e_x_k + W_e_h_s)
                attention_tanh = tf.transpose(attention_tanh, [0, 1, 3, 2])
                alpha_t = tf.nn.conv2d(attention_tanh, V_e, strides=[1, 1, 1, 1], padding='VALID')
                alpha_t = tf.reshape(alpha_t, [-1, 1, self.inp_features])
                alpha_t = tf.nn.softmax(alpha_t)
                output = tf.reshape(inputs * alpha_t, [-1, self.inp_features] )
                # output is ~x
                return output

    def Encoder(self, feature_inp, num_hidden, drop_out_rate):
        attention_input = feature_inp
        encoder_inputs = tf.split(attention_input, self.input_time_step, 1)
        with tf.variable_scope('Encoder'):
            # create  network
            cells = []
            for i in range(self.num_layers):
                cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
                if drop_out_rate:
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1 - drop_out_rate)
                cells.append(cell)
            encoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            batch_size = array_ops.shape(encoder_inputs[0])[0]
            state = encoder_cell.zero_state(batch_size, dtype=tf.float32)
            encoder_output = []
            # encoder
            for i, input_per_time in enumerate(encoder_inputs):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                encoder_attention_input = self.spatial_attention(input_per_time, attention_input, state)
                output, state = encoder_cell(encoder_attention_input, state)
                output = tf.reshape(output, [-1, 1, output.get_shape()[-1].value])
                encoder_output.append(output)
                # encoder_output is h
        return encoder_output, state

    def temporal_attention(self, inputs, last_state, num_hidden):
        """
        Compute spatial attention given inputs and attention inputs
        :param inputs: tensor, shape:[batch_size, 1, num_features]
        :param last_state:  shape: list, contains the states of the number of layers.
        :param num_hidden:  number of layers.
        :return: A tensor multiplied by the attention weights, shape: [batch_size, 1, num_features]
        """
        with tf.variable_scope('temporal_attention'):
            # attention input is h
            last_state = nest.flatten(last_state)
            last_state = tf.concat(last_state, axis=1)
            last_state = tf.expand_dims(last_state, axis=-1)
            last_state = tf.concat([last_state for _ in range(self.input_time_step)], axis=2)
            dim = last_state.get_shape()[1].value
            U_d = tf.get_variable(shape=[1, num_hidden, 1, num_hidden], name='U_d')
            W_d = tf.get_variable(shape=[1, dim, 1, num_hidden], name='W_d')
            V_d = tf.get_variable(shape=[1, num_hidden, 1, 1], name='V_d')
            attention_input = tf.concat(inputs, axis=1)
            attention_input = tf.transpose(attention_input, [0, 2, 1])
            U_d_h_i = self.hd_matmul(U_d, attention_input)
            W_d_h_s = self.hd_matmul(W_d, last_state)
            attention_tanh = tf.nn.tanh(W_d_h_s + U_d_h_i)
            attention_tanh = tf.transpose(attention_tanh, [0, 3, 2, 1])
            alpha_t = tf.nn.conv2d(attention_tanh, V_d, strides=[1, 1, 1, 1], padding='VALID')
            alpha_t = tf.reshape(alpha_t, [-1, 1, self.input_time_step])
            alpha_t = tf.nn.softmax(alpha_t)
            output = attention_input * alpha_t
            output = tf.reduce_sum(output, axis=2)
            # output is ct
            return output

    def Decoder(self, decoder_inputs, state, num_hidden, drop_out_rate):
        with tf.variable_scope('Decoder', reuse=tf.AUTO_REUSE):
            # create network
            cells = []
            for i in range(self.num_layers):
                cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
                if drop_out_rate:
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1-drop_out_rate)
                cells.append(cell)
            decoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            # decoder
            batch_size = array_ops.shape(decoder_inputs[0])[0]
            decoder_output = []
            for i in range(self.output_time_step):
                if i == 0:
                    y_t_1 = tf.zeros([batch_size, 1], dtype=tf.float32)
                c_t = self.temporal_attention(decoder_inputs, state, num_hidden)
                decoder_attention_input = tf.concat([y_t_1, c_t], axis=1)
                W_tilta = tf.get_variable(name='W_tilta', shape=[decoder_attention_input.shape[-1].value, 1], dtype=tf.float32)
                b = tf.get_variable(name='b', shape=1, dtype=tf.float32)
                y_t_1_tilta = tf.matmul(decoder_attention_input, W_tilta) + b
                # state_tmp = h
                state_tmp = []
                for s in state:
                    state_tmp.append(s[1])
                state_tmp = tf.concat(state_tmp, axis=1)
                decoder_input = tf.concat([state_tmp, y_t_1_tilta], axis=1)
                d_t, state = decoder_cell(decoder_input, state)
                p = d_t.get_shape()[1].value
                d_t_c_t = tf.concat([d_t, c_t], axis=1)
                b_w = tf.get_variable(name='b_w', shape=p, dtype=tf.float32)
                b_v = tf.get_variable(name='b_v', shape=1, dtype=tf.float32)
                m = num_hidden
                p_m = p + m
                W_y = tf.get_variable(name='W_y', shape=[p_m, p])
                v_y = tf.get_variable(name='V_y', shape=[p, 1])
                y_t = tf.matmul((tf.matmul(d_t_c_t, W_y) + b_w), v_y) + b_v
                y_t_1 = y_t
                yt = tf.squeeze(y_t, axis=1)
                decoder_output.append(yt)
            return decoder_output
