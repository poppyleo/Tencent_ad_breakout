import tensorflow as tf

def attention_layer(self, inputs, attention_size, time_major=False, attention_mask=False, return_alphas=False,
                    name="attention_pooling"):
    """
     Attention mechanism layer which reduces Bi-LSTM outputs with Attention vector.
    :param inputs:
    :param attention_size:
    :param time_major:
    :param attention_mask:
    :param return_alphas:
    :param name:
    :return:
    The Attention output `Tensor`.
    In case of RNN, this will be a `Tensor` shaped:
        `[batch_size, cell.output_size]`.
    In case of Bidirectional RNN, this will be a `Tensor` shaped:
        `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """
    with tf.variable_scope(name):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        if isinstance(inputs, tuple):
            inputs = tf.concat(inputs, 2)
        # (T,B,D) => (B,T,D)
        if time_major:
            inputs = tf.array_ops.transpose(inputs, [1, 0, 2])
        # T, D value
        sequence_length = inputs.get_shape()[1].value
        hidden_size = inputs.shape[2].value
        # Trainable parameters
        w_omega = tf.get_variable('w_omega', shape=[hidden_size, attention_size])
        b_omega = tf.get_variable('b_omega', shape=[attention_size])
        u_omega = tf.get_variable('u_omega', shape=[attention_size])
        # (B*T,D)*(D,A)+(1,A)=(B*T,A)
        v = tf.matmul(tf.reshape(inputs, [-1, hidden_size]), w_omega) + tf.reshape(b_omega, [1, -1])
        v = tf.tanh(v)
        # (B*T,A)*(A,1)=(B,T,1) -> (B,T)
        vu = tf.reshape(tf.matmul(v, tf.reshape(u_omega, [-1, 1])), [-1, sequence_length])
        # mask the _pad_ token with zero
        if attention_mask:
            vu += (1.0 - tf.cast(self.input_x_mask, tf.float32)) * -10000
        alphas = tf.nn.softmax(vu, name='alphas')
        # (B,T,D) . (B,T,1) = (B,T,A) -> (B,A)
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), axis=1)
    if return_alphas:
        return output, alphas
    else:
        return output
