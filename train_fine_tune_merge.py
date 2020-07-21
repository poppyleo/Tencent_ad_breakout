import os
import time
import json
import tqdm
from config import Config
from model import *
from utils import DataIterator
from optimization import create_optimizer
import numpy as np
from bert import tokenization
from sklearn.metrics import  accuracy_score
from keras.optimizers import Adam

"""
做20分类
#dym+mean decay_step10000:/home/none404/hm/Tencent_ads/finetune_data/all_model/runs_7/1590680486 max_len 128
#dym+mean decay_step10000:/home/none404/hm/Tencent_ads/finetune_data/all_model/runs_3/1590723121 max_len 256


bert_256 clean-data dym+avg[0.85,0.15]fold_1:/home/none404/hm/Tencent_ads/finetune_data/all_model/runs_7/1591548253 #1.434

"""

gpu_id = 7
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
result_data_dir = Config().data_processed
print('GPU ID: ', str(gpu_id))
print('Fine Tune Learning Rate: ', Config().embed_learning_rate)
print('Pretrained Model Vocab: ', Config().vocab_file)
print('Batch Size: ', Config().batch_size)
print('Use original bert', Config().use_origin_bert)
print('Use avg pool', Config().is_avg_pool)
print('loss name:', Config().loss_name)
print('weight:', Config().joint_rate)


def softmax(x, axis=1):
    """
    自写函数定义softmax
    :param x:
    :param axis:
    :return:
    """
    # 计算每行的最大值
    row_max = x.max(axis=axis)

    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    row_max=row_max.reshape(-1, 1)
    x = x - row_max
    #计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s

def cacu_acc(true_merge_list, pred_merge_list, mapping_dict):
    true_merge_list = [mapping_dict[i] for i in true_merge_list]
    pred_merge_list = [mapping_dict[i] for i in pred_merge_list]

    true_gender_list = [int(i[0]) for i in true_merge_list]
    true_age_list = [int(i[1:]) for i in true_merge_list]

    pred_gender_list= [int(i[0]) for i in pred_merge_list]
    pred_age_list= [int(i[1:]) for i in pred_merge_list]

    age_auc = accuracy_score(true_age_list, pred_age_list)
    gender_auc = accuracy_score(true_gender_list, pred_gender_list)
    return age_auc, gender_auc

def train(train_iter, test_iter, config):
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session = tf.Session(config=session_conf)
        with session.as_default():
            model = Model(config)  # config.sequence_length,

            global_step = tf.Variable(0, name='step', trainable=False)
            learning_rate = tf.train.exponential_decay(config.learning_rate, global_step, config.decay_step,
                                                       config.decay_rate, staircase=True)

            normal_optimizer = tf.train.AdamOptimizer(learning_rate)

            all_variables = graph.get_collection('trainable_variables')
            bert_var_list = [x for x in all_variables if 'bert' in x.name]
            normal_var_list = [x for x in all_variables if 'bert' not in x.name]
            print('bert train variable num: {}'.format(len(bert_var_list)))
            print('normal train variable num: {}'.format(len(normal_var_list)))
            normal_op = normal_optimizer.minimize(model.loss, global_step=global_step, var_list=normal_var_list)
            num_batch = int(train_iter.num_records / config.batch_size * config.train_epoch)
            embed_step = tf.Variable(0, name='step', trainable=False)
            if bert_var_list:  # 对bert微调
                print('bert trainable!!')
                word2vec_op, embed_learning_rate, embed_step = create_optimizer(
                    model.loss, config.embed_learning_rate, num_train_steps=num_batch,
                    num_warmup_steps=int(num_batch * 0.05) , use_tpu=False ,  variable_list=bert_var_list
                )

                train_op = tf.group(normal_op, word2vec_op)
            else:
                train_op = normal_op

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(
                os.path.join(config.save_model, "runs_" + str(gpu_id), timestamp))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            with open(out_dir + '/' + 'config.json', 'w', encoding='utf-8') as file:
                json.dump(config.__dict__, file)
            print("Writing to {}\n".format(out_dir))

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=config.num_checkpoints)
            if config.continue_training:
                print('recover from: {}'.format(config.checkpoint_path))
                saver.restore(session, config.checkpoint_path)
            else:
                session.run(tf.global_variables_initializer())
            cum_step = 0
            for i in range(config.train_epoch):
                for input_ids_list, input_mask_list, segment_ids_list,age_list,gender_list,all_label_list, seq_length in tqdm.tqdm(
                        train_iter):
                    feed_dict = {
                        model.input_x_word: input_ids_list,
                        model.input_mask: input_mask_list,
                        model.age: age_list,
                        model.gender: gender_list,
                        model.input_x_len:seq_length,
                        model.merge_label : all_label_list,
                        model.keep_prob: config.keep_prob,
                        model.is_training: True,
                    }

                    _, step, _, loss, lr = session.run(
                            fetches=[train_op,
                                     global_step,
                                     embed_step,
                                     model.loss,
                                     learning_rate
                                     ],
                            feed_dict=feed_dict)

                    if cum_step % 10 == 0:
                        format_str = 'step {}, loss {:.4f} lr {:.5f}'
                        print(
                            format_str.format(
                                step, loss, lr)
                        )
                    cum_step += 1

                age_auc, gender_auc = set_test(model, test_iter, session)
                print('dev set : cum_step_{},age_auc_{},gender_auc_{}'.format(cum_step, age_auc, gender_auc))

                saver.save(session, os.path.join(out_dir, 'model_{:.4f}_{:.4f}'.format(age_auc, gender_auc)), global_step=step)


def set_test(model, test_iter, session):

    if not test_iter.is_test:
        test_iter.is_test = True

    true_merge_list = []
    pred_merge_list = []
    for input_ids_list, input_mask_list, segment_ids_list,age_list,gender_list,all_label_list, seq_length in tqdm.tqdm(
            test_iter):

        feed_dict = {
            model.input_x_word: input_ids_list,
            model.input_x_len: seq_length,
            model.age: age_list,
            model.gender: gender_list,
            model.merge_label:all_label_list,
            model.input_mask: input_mask_list,

            model.keep_prob: 1,
            model.is_training: False,
        }

        merge_logits = session.run(
            fetches=[model.merge_logits],
            feed_dict=feed_dict
        )[0]

        merge_label=softmax(merge_logits)
        merge_label = np.argmax(merge_label, axis=1)

        true_merge_list.extend(all_label_list)
        pred_merge_list.extend(merge_label)
    assert len(true_merge_list)==len(pred_merge_list)
    save_dict = config.save_dict
    mapping_dict =dict([(j, i) for i, j in save_dict.items()])
    age_auc, gender_auc = cacu_acc(true_merge_list,pred_merge_list,mapping_dict)
    print('focal_auc {}, age_auc {}, gender_auc {}'.format(age_auc+gender_auc, age_auc,gender_auc))
    return age_auc, gender_auc


if __name__ == '__main__':
    config = Config()
    vocab_file = config.vocab_file  # 通用词典
    do_lower_case = False
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    train_iter = DataIterator(config.batch_size, data_file=config.data_processed + 'new_train.csv',
                              use_bert=config.use_bert,
                              tokenizer=tokenizer, seq_length=config.sequence_length)

    dev_iter = DataIterator(config.batch_size, data_file=config.data_processed + 'new_dev.csv',
                            use_bert=config.use_bert, tokenizer=tokenizer,
                            seq_length=config.sequence_length, is_test=True)

    train(train_iter, dev_iter, config)
