from config import Config
import tensorflow as tf
import os
import json
import numpy as np
from bert import tokenization
import tqdm
import pandas as pd
from utils import DataIterator
import pickle


result_data_dir = Config().result
gpu_id = 6
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
print('GPU ID: ', str(gpu_id))
print('Data dir: ', result_data_dir)
print('Pretrained Model Vocab: ', Config().vocab_file)
print('Model: ', Config().checkpoint_path)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


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


def get_session(checkpoint_path):
    graph = tf.Graph()

    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session = tf.Session(config=session_conf)
        with session.as_default():
            # Load the saved meta graph and restore variables
            try:
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_path))
            except OSError:
                saver = tf.train.import_meta_graph("{}.ckpt.meta".format(checkpoint_path))
            saver.restore(session, checkpoint_path)

            _input_x = graph.get_operation_by_name("input_x_word").outputs[0]
            _input_x_len = graph.get_operation_by_name("input_x_len").outputs[0]
            _input_mask = graph.get_operation_by_name("input_mask").outputs[0]
            _keep_ratio = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
            _is_training = graph.get_operation_by_name('is_training').outputs[0]

            age_logits = graph.get_tensor_by_name('age_relation/age_relation_logits/BiasAdd:0')
            gender_logits = graph.get_tensor_by_name('gender_relation/gender_relation_logits/BiasAdd:0')

            def run_predict(feed_dict):
                return session.run([age_logits, gender_logits], feed_dict)

    print('recover from: {}'.format(checkpoint_path))
    return run_predict, (_input_x, _input_x_len, _input_mask, _keep_ratio, _is_training)


def set_test(test_iter, model_file):
    if not test_iter.is_test:
        test_iter.is_test = True

    pred_age_list = []
    pred_gender_list = []
    pred_gender_logits =[]
    pred_age_logits =[]
    predict_fun, feed_keys = get_session(model_file)
    for input_ids_list, input_mask_list, segment_ids_list,age_list,gender_list,all_label_list, seq_length in tqdm.tqdm(test_iter):

        age_logits, gender_logits = predict_fun(
            dict(
                zip(feed_keys, (input_ids_list, seq_length, input_mask_list, 1, False))
                 )
        )
        age_logits=softmax(age_logits)
        gender_logits=softmax(gender_logits)

        age_label = np.argmax(age_logits, axis=1) + 1  #还原
        gender_label = np.argmax(gender_logits, axis=1) +1 #还原

        pred_age_list.extend(age_label)
        pred_gender_list.extend(gender_label)

        pred_gender_logits.extend(gender_logits)
        pred_age_logits.extend(age_logits)

    assert len(pred_age_list) == len(pred_gender_list)

    # test_df = pd.read_csv(config.data_processed + 'new_test_{}_{}.csv'.format(start,end))
    test_df = pd.read_csv(config.data_processed + 'new_test.csv')

    predict_df = pd.DataFrame()
    predict_df['user_id']= test_df['user_id']
    predict_df['predicted_age'] = pred_age_list
    predict_df['predicted_gender'] = pred_gender_list
    # predict_df['predicted_gender'] = 0


    model_name = config.checkpoint_path.split('/')[-1]
    print(model_name)
    save_p=result_data_dir + model_name
    if not os.path.exists(save_p):
        os.mkdir(save_p)
    #
    # with open(save_p+'/age_logits.pickle_{}_{}'.format(start,end),'wb') as f:
    #     pickle.dump(pred_age_logits,f)
    # with open(save_p+'/gender_logits.pickle_{}_{}'.format(start,end),'wb') as f:
    #     pickle.dump(pred_gender_logits,f)
    with open(save_p+'/age_logits.pickle','wb') as f:
        pickle.dump(pred_age_logits,f)
    with open(save_p+'/gender_logits.pickle','wb') as f:
        pickle.dump(pred_gender_logits,f)

    # predict_df.to_csv(save_p +'/predict_{}_{}.csv'.format(start,end), encoding='utf-8',index=False)
    predict_df.to_csv(save_p +'/{}_predict.csv'.format(model_name), encoding='utf-8',index=False)
    """
       融合所需参数保存
    """


if __name__ == '__main__':
    config = Config()
    vocab_file= config.vocab_file
    do_lower_case =False
    jump =7
    start,end = (jump-1)*200000,jump*200000
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    print('Predicting test.txt..........')
    print('Jump：',jump)

    # dev_iter = DataIterator(config.test_batch_size, data_file=config.data_processed + 'new_test_{}_{}.csv'.format(start,end),
    #                         use_bert=config.use_bert,seq_length=config.sequence_length, is_test=True, tokenizer=tokenizer)
    dev_iter = DataIterator(config.test_batch_size, data_file=config.data_processed + 'new_test.csv',
                            use_bert=config.use_bert,seq_length=config.sequence_length, is_test=True, tokenizer=tokenizer)

    set_test(dev_iter, config.checkpoint_path)
