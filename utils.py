import numpy as np
from bert import tokenization
from tqdm import tqdm
from config import Config
import pandas as pd
import os
gpu_id = 5
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

def load_data(data_file):
    """
    读取数据
    :param file:
    :return:
    """
    data_df = pd.read_csv(data_file)
    data_df.fillna('', inplace=True)
    lines = list(zip(list(data_df['text']), list(data_df['age']),list(data_df['gender']),list(data_df['all_label'])))

    return lines


def create_example(lines):
    examples = []
    for (_i, line) in enumerate(lines):
        text = str(line[0])
        age = int(line[1])
        gender = int(line[2])
        all_label = int(line[3])
        examples.append(InputExample(text=text,age=age,gender=gender,all_label=all_label))
    return examples


def get_examples(data_file):
    return create_example(
        load_data(data_file)
    )


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self,text,age,gender,all_label):
        self.text = text
        self.age = age
        self.gender = gender
        self.all_label = all_label


class DataIterator:
    """
    数据迭代器
    """
    def __init__(self, batch_size, data_file, tokenizer, use_bert=False, seq_length=100, is_test=False,):
        self.data_file = data_file
        self.data = get_examples(data_file)
        self.batch_size = batch_size
        self.use_bert = use_bert
        self.seq_length = seq_length
        self.num_records = len(self.data)
        self.all_tags = []
        self.idx = 0  # 数据索引
        self.all_idx = list(range(self.num_records))  # 全体数据索引
        self.is_test = is_test

        if not self.is_test:
            self.shuffle()
        self.tokenizer = tokenizer
        print(self.num_records)

    def convert_single_example(self, example_idx):
        text = self.data[example_idx].text.split(' ')
        age = self.data[example_idx].age - 1
        gender = self.data[example_idx].gender - 1
        all_label =self.data[example_idx].all_label

        q_tokens = []
        ntokens = []
        segment_ids = []

        """得到input的token-----start-------"""

        ntokens.append("[CLS]")
        segment_ids.append(0)
        """text"""
        # 得到问题的token
        for i, word in enumerate(text):
            token = self.tokenizer.tokenize(word)
            q_tokens.extend(token)
        # 把问题的token加入至所有字的token中
        for i, token in enumerate(q_tokens):
            ntokens.append(token)
            segment_ids.append(0)

        # 长于MAX LEN 则截断
        if ntokens.__len__() >= self.seq_length - 1:
            ntokens = ntokens[:(self.seq_length - 1)]
            segment_ids = segment_ids[:(self.seq_length - 1)]

        ntokens.append("[SEP]")
        segment_ids.append(1)

        """得到input的token-------end--------"""

        """token2id---start---"""
        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        # label_mask = [1] * len(input_ids)
        while len(input_ids) < self.seq_length:
            # 不足时补零
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # we don't concerned about it!
            ntokens.append("**NULL**")
            # label_mask.append(0)
        assert len(input_ids) == self.seq_length
        assert len(input_mask) == self.seq_length
        assert len(segment_ids) == self.seq_length
        """token2id ---end---"""
        return input_ids, input_mask, segment_ids, age,gender,all_label

    def shuffle(self):
        np.random.shuffle(self.all_idx)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.num_records:  # 迭代停止条件
            self.idx = 0
            if not self.is_test:
                self.shuffle()
            raise StopIteration

        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        age_list = []
        gender_list = []
        all_label_list = []

        num_tags = 0

        while num_tags < self.batch_size:  # 每次返回batch_size个数据
            idx = self.all_idx[self.idx]
            res = self.convert_single_example(idx)
            if res is None:
                self.idx += 1
                if self.idx >= self.num_records:
                    break
                continue
            input_ids, input_mask, segment_ids, age,gender,all_label = res

            # 一个batch的输入
            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            all_label_list.append(all_label)
            age_list.append(age)
            gender_list.append(gender)
            if self.use_bert:
                num_tags += 1

            self.idx += 1
            if self.idx >= self.num_records:
                break

        return input_ids_list, input_mask_list, segment_ids_list,age_list,gender_list,all_label_list, self.seq_length


if __name__ == '__main__':
    config = Config()
    vocab_file = config.vocab_file
    do_lower_case = True
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    str_flag = config.train_type
    # print(vocab_file)
    # print(print(len(tokenizer.vocab)))

    # data_iter = DataIterator(config.batch_size, data_file= config.dir_with_mission + 'train.txt', use_bert=True,
    #                         seq_length=config.sequence_length, tokenizer=tokenizer)
    #
    # dev_iter = DataIterator(config.batch_size, data_file=config.dir_with_mission + 'dev.txt', use_bert=True,
    #                          seq_length=config.sequence_length, tokenizer=tokenizer, is_test=True)
    train_iter = DataIterator(config.batch_size,
                              data_file=config.data_processed+ 'new_train.csv',
                              use_bert=config.use_bert,
                              tokenizer=tokenizer, seq_length=config.sequence_length)
    for input_ids_list, input_mask_list, segment_ids_list,age_list,gender_list,all_label_list, seq_length in tqdm(train_iter):
        for i in input_ids_list:
            print(i)
            print()
        # print(input_ids_list)
        # print(segment_ids_list[0])
        # print(input_mask_list[0])
        # print(age_list)
        # print(gender_list)
        # print(all_label_list)
        break



