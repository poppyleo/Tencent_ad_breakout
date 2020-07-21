import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from config import Config
import os
from tqdm import tqdm
from multiprocessing import Pool
import csv
config = Config()

def remove_tube(x):
    text_list = x.split(' ')
    tube_list = [str(i+1) for i in range(256)]
    pad_tube = [tube for tube in tube_list if tube in text_list]
    for pad_ in pad_tube:
        while True:
            try:
                text_list.remove(pad_)
            except:
                break
    x = ' '.join(text_list)
    return x

if __name__ == '__main__':
    str_flag = config.train_type
    print(str_flag)
    file_path = config.finetune_data + '{}_data/'.format(str_flag)

    merge_path = file_path + 'merge_clean.csv'
    if not  os.path.exists(merge_path):
        train_df = pd.read_csv(file_path + 'train.csv')
        test_df = pd.read_csv(file_path + 'test.csv')
        merge_df = train_df.append(test_df)
        merge_df['text'] = merge_df['text'].apply(remove_tube)
    else:
        merge_df = pd.read_csv(file_path + 'merge_clean.csv')


    #计算TF-IDF的值
    print('Compute TF-IDF......')
    corpus = merge_df ['text']
    user_list = merge_df ['user_id']
    tfidf_vec = TfidfVectorizer(min_df=256,max_df=0.90)  #出现小于256次的不要，去掉90%文档都出线过的文档
    tfidf_matrix = tfidf_vec.fit_transform(corpus)
    Essence_word = np.array(tfidf_vec.get_feature_names())

    result_list = []
    for row  in  tqdm(corpus[:100]):
        new_text = []
        for word in row.split(' '):
            if word=='隔':
                new_text.append(word)
            elif Essence_word.__contains__(word):
                new_text.append(word)
            else:
                pass
        result_list.append(' '.join(new_text))
    merge_df['text'] = result_list
    merge_df.to_csv(file_path + 'merge_clean.csv',index=False)

    # tfidf_matrix = tfidf_vec.fit_transform(corpus)
    # corpus_vec = tfidf_matrix.toarray()
    # f1 = open(config.data_processed + 'all_finetune_tf.csv', 'w+')
    # writer_1 = csv.writer(f1)
    # writer_1.writerow(["user_id", "text", "age",'gender'])
    # p = Pool(64)
    # merge_path = file_path + 'merge_clean.csv'
    # if not  os.path.exists(merge_path):
    #     train_df = pd.read_csv(file_path + 'train.csv')
    #     test_df = pd.read_csv(file_path + 'test.csv')
    #     merge_df = train_df.append(test_df)
    #     merge_df['text'] = merge_df['text'].apply(remove_tube)
    # else:
    #     merge_df = pd.read_csv(file_path + 'merge_clean.csv')
    #
    # print('Compute TF-IDF......')
    # corpus = merge_df ['text']
    # user_list = merge_df ['user_id']








"""
 #计算TF-IDF的值
    import json
    from gensim.models import TfidfModel
    from gensim.corpora import Dictionary
    from itertools import islice


    def extract_keywords(dct, tfidf, threshold=0.2, topk=5):
        '''
        dct :: Dictionary
        tfidf :: model[doc], [(int, number)]
        threshold :: 提取tftdf值超过它的词
        topk :: 最多提取个数
        '''
        tfidf = sorted(tfidf, key=lambda x: x[1], reverse=True)
        return list(islice([dct[w] for w, score in tfidf if score > threshold], topk))
    if not os.path.exists(file_path + '/news_tfidf.model'):
        data = [doc.split() for doc in corpus]
        dct = Dictionary(data)
        corpus = [dct.doc2bow(doc) for doc in data]  # convert corpus to BoW format
        model = TfidfModel(corpus)  # fit model
        dct.save(file_path+'/news.dict')
        model.save(file_path+'/news_tfidf.model')
    else:
        dct = Dictionary.load('/news.dict')
        data = [doc.split() for doc in corpus]
        corpus = [dct.doc2bow(doc) for doc in data]
        model = TfidfModel.load('/news_tfidf.model')

    for text, doc in zip(corpus, data):
        print(extract_keywords(dct, model[doc]))
        print()
        break
"""

