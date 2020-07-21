import tensorflow as tf
from config import Config
config = Config()
from tf_utils.bert_modeling import BertModel, BertConfig, get_assignment_map_from_checkpoint, get_shape_list  # BERT
# from tf_utils.electra_modeling import BertModel, BertConfig, get_assignment_map_from_checkpoint, get_shape_list  # electra
from tensorflow.contrib.layers.python.layers import initializers
from tf_utils.crf_utils import rnncell as rnn
from tf_utils.focal_loss import focal_loss
# import configure_finetuning
# import memory_saving_gradients
# tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory
# 对于CRF这种多优化目标的层，memory_saving_gradients会出bug，注释即可。


def kl_for_log_probs(log_p, log_q):
    p = tf.exp(log_p)
    neg_ent = tf.reduce_sum(p * log_p, axis=-1)
    neg_cross_ent = tf.reduce_sum(p * log_q, axis=-1)
    kl = neg_ent - neg_cross_ent
    return kl


def hidden_to_logits(hidden, is_training, num_classes, scope):
    hidden_size = hidden.shape[-1].value
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        output_weights = tf.get_variable(
            "output_weights", [num_classes, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_classes], initializer=tf.zeros_initializer())

        if is_training:
            # I.e., 0.1 dropout
            hidden = tf.nn.dropout(hidden, keep_prob=0.9)

        if hidden.shape.ndims == 3:
            logits = tf.einsum("bid,nd->bin", hidden, output_weights)
        else:
            logits = tf.einsum("bd,nd->bn", hidden, output_weights)
        logits = tf.nn.bias_add(logits, output_bias)

    return logits

class Model:

    def __init__(self, config):
        self.config = config
        self.input_x_word = tf.placeholder(tf.int32, [None, None], name="input_x_word")
        self.input_x_len = tf.placeholder(tf.int32, name='input_x_len')
        self.input_mask = tf.placeholder(tf.int32, [None, None], name='input_mask')
        self.age = tf.placeholder(tf.int32, [None], name='age')  # 年龄标签
        self.gender = tf.placeholder(tf.int32, [None], name='gender')  # 性别标签
        self.merge_label = tf.placeholder(tf.int32, [None], name='merge_label')  # 20分类标签
        self.keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.is_training = tf.placeholder(tf.bool, None, name='is_training')
        self.initializer = initializers.xavier_initializer()

        self.age_num = config.age_num
        self.gender_num = config.gender_num
        self.merge_num = self.age_num*self.gender_num
        self.model_type = config.model_type

        print('Run Model Type:', self.model_type)

        self.init_embedding(bert_init=True)
        output_layer = self.word_embedding

        # some hyper_parameters
        used = tf.sign(tf.abs(self.input_x_word))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.input_x_word)[0]
        self.num_steps = tf.shape(self.input_x_word)[-1]

        if self.model_type == 'bilstm':
            lstm_inputs = tf.nn.dropout(output_layer, config.dropout)
            # bi-directional lstm layer
            model_outputs = self.biLSTM_layer(lstm_inputs, config.lstm_dim, self.lengths)  # lstm_dim = 100
            pool_size = self.config.sequence_length
            hidden_size = model_outputs.shape[-1]

        elif self.model_type == 'gru':
            print(self.model_type)
            gru_inputs = tf.nn.dropout(output_layer, config.dropout)
            # bi-directional gru layer
            GRU_cell_fw = tf.contrib.rnn.GRUCell(config.gru_num)  # 参数可调试
            # 后向
            GRU_cell_bw = tf.contrib.rnn.GRUCell(config.gru_num)  # 参数可调试
            output_layer_1 = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                             cell_bw=GRU_cell_bw,
                                                             inputs=gru_inputs,
                                                             sequence_length=None,
                                                             dtype=tf.float32)[0]
            output_layer_1 = tf.concat([output_layer_1[0], output_layer_1[1]], axis=-1)
            model_outputs = output_layer_1
            pool_size = config.sequence_length
            hidden_size = model_outputs.shape[-1]

        else:  # only bert_output
            model_outputs = output_layer
            pool_size = self.config.sequence_length
            hidden_size = get_shape_list(output_layer)[-1]

        # 池化+drop_out
        if self.config.is_avg_pool:
            print('is_avg_pool:', self.config.is_avg_pool)
            print(self.model_type)
            output_layer = model_outputs
            # avpooled_out = tf.layers.max_pooling1d(output_layer, pool_size=pool_size, strides=1)
            avpooled_out = tf.layers.average_pooling1d(output_layer, pool_size=pool_size, strides=1)  # shape = [batch, hidden_size]
            print(avpooled_out.shape)
            avpooled_out = tf.reshape(avpooled_out, [-1, hidden_size])

            cls_out = output_layer[:, 0:1, :]  # pooled_output
            cls_out = tf.squeeze(cls_out, axis=1)

        else:
            print('CLS:', True)
            avpooled_out = output_layer[:, 0:1, :]  # pooled_output
            avpooled_out = tf.squeeze(avpooled_out, axis=1)

        def logits_and_predict(avpooled_out,num_classes, name_scope=None):
            with tf.name_scope(name_scope):
                inputs = tf.nn.dropout(avpooled_out, keep_prob=self.keep_prob)  # delete dropout
                logits = tf.layers.dense(inputs, num_classes,
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                         name=name_scope+'_logits')
                predict = tf.round(tf.sigmoid(logits), name=name_scope+"predict")

            return logits, predict

        """性别年龄是否用同一条向量"""
        if config.joint:
            #联合学习任务
            avpooled_out_1= avpooled_out
            avpooled_out_2= avpooled_out
            # avpooled_out_2= cls_out
            self.age_logits, self.age_predict = logits_and_predict(avpooled_out_1,self.age_num, name_scope='age_relation')
            age_one_hot_labels = tf.one_hot(self.age, depth=self.age_num, dtype=tf.float32)

            self.gender_logits, self.gender_predict = logits_and_predict(avpooled_out_2,self.gender_num, name_scope='gender_relation')
            gender_one_hot_labels = tf.one_hot(self.gender, depth=self.gender_num, dtype=tf.float32)
            if config.loss_name == 'focal_loss':
                age_loss = focal_loss(self.age_logits,self.age,self.config.batch_size
                                      ,alpha=[[0.35195],[1.49271],[2.02909],[1.50578],[1.30667],[1.01720],[0.66711],[0.31967],[0.19474],[0.11508]],multi_dim=False)
                gender_loss = focal_loss(self.gender_logits,self.gender,self.config.batch_size,alpha=[[2],[1]],multi_dim=False)
            else:
                age_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=age_one_hot_labels, logits=self.age_logits)
                age_loss = tf.reduce_mean(tf.reduce_sum(age_losses, axis=1))

                gender_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=gender_one_hot_labels, logits=self.gender_logits)
                gender_loss = tf.reduce_mean(tf.reduce_sum(gender_losses, axis=1))
            self.loss = config.joint_rate[0]*age_loss + config.joint_rate[1]*gender_loss
        else:
            #20分类
            self.merge_logits, self.merge_predict = logits_and_predict(avpooled_out,self.merge_num, name_scope='merge_relation')
            merge_one_hot_labels = tf.one_hot(self.merge_label, depth=self.merge_num, dtype=tf.float32)
            merge_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=merge_one_hot_labels, logits=self.merge_logits)
            merge_loss = tf.reduce_mean(tf.reduce_sum(merge_losses, axis=1))
            self.loss = merge_loss


    def biLSTM_layer(self, lstm_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, 2*lstm_dim]
        """
        with tf.name_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.name_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                lstm_inputs,
                dtype=tf.float32,
                sequence_length=lengths)
        return tf.concat(outputs, axis=2)

    def init_embedding(self, bert_init=True):
        with tf.name_scope('embedding'):
            word_embedding = self.bert_embed(bert_init)
            print('self.config.embed_dense_dim:', self.config.embed_dense_dim)
            word_embedding = tf.layers.dense(word_embedding, self.config.embed_dense_dim, activation=tf.nn.relu)
            hidden_size = word_embedding.shape[-1].value
        self.word_embedding = word_embedding
        print(word_embedding.shape)
        self.output_layer_hidden_size = hidden_size

    def bert_embed(self, bert_init=True):
        """加载bert模型结构"""
        bert_config_file = self.config.bert_config_file
        bert_config = BertConfig.from_json_file(bert_config_file)
        model = BertModel(
            config=bert_config,
            is_training=self.is_training,  # 微调
            input_ids=self.input_x_word,
            input_mask=self.input_mask,
            token_type_ids=None,
            use_one_hot_embeddings=False)

        layer_logits = []
        for i, layer in enumerate(model.all_encoder_layers):
            layer_logits.append(
                tf.layers.dense(
                    layer, 1,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                    name="layer_logit%d" % i
                )
            )

        layer_logits = tf.concat(layer_logits, axis=2)  # 第三维度拼接
        layer_dist = tf.nn.softmax(layer_logits)
        seq_out = tf.concat([tf.expand_dims(x, axis=2) for x in model.all_encoder_layers], axis=2)
        pooled_output = tf.matmul(tf.expand_dims(layer_dist, axis=2), seq_out)
        pooled_output = tf.squeeze(pooled_output, axis=2)
        pooled_layer = pooled_output

        char_bert_outputs = pooled_layer

        if self.config.pretrainning_model=='electra':
            if self.config.use_origin_bert:
                final_hidden_states = model.get_sequence_output()  # 原生elctra
                self.config.embed_dense_dim = 256
            else:
                final_hidden_states = char_bert_outputs  # 多层融合elctra
                self.config.embed_dense_dim = 256
        else:
            if self.config.use_origin_bert:
                final_hidden_states = model.get_sequence_output()  # 原生bert
                self.config.embed_dense_dim = 768
            else:
                final_hidden_states = char_bert_outputs  # 多层融合bert
                self.config.embed_dense_dim = 512

        tvars = tf.trainable_variables()
        init_checkpoint = self.config.bert_file  # './chinese_L-12_H-768_A-12/bert_model.ckpt'
        assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        if bert_init:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            print("  name = {}, shape = {}{}".format(var.name, var.shape, init_string))
        print('init bert from checkpoint: {}'.format(init_checkpoint))
        return final_hidden_states
