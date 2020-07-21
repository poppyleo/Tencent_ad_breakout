class Config:
    
    def __init__(self):

        self.embed_dense = True
        self.embed_dense_dim = 512
        self.warmup_proportion = 0.05
        self.use_bert = True
        self.keep_prob = 0.8
        self.over_sample = True


        self.num_checkpoints = 20 * 3

        self.train_epoch = 3
        self.sequence_length = 256

        self.learning_rate = 1e-4
        self.embed_learning_rate = 3e-5

        self.batch_size = 46
        self.test_batch_size = 148#测试集batch_size
        self.decay_rate = 0.5
        self.decay_step = int(720000/self.batch_size)
        self.embed_trainable = True

        self.as_encoder = True
        self.age_num = 10
        self.gender_num = 2

        # predict.py ensemble.py get_ensemble_final_result.py post_ensemble_final_result.py的结果路径
        self.continue_training = False
        self.ensemble_source_file = '/data/wangzhili/nCoV/processed_data/ensemble/source_file/'
        self.ensemble_result_file = '/data/wangzhili/nCoV/processed_data/ensemble/result_file/'

        # """联合"""
        # self.checkpoint_path = "/home/none404/hm/Tencent_ads/finetune_data/all_model/runs_5/1590595076/model_0.4332_0.9313-13000"





        #  数据预处理的路径
        #path
        self.pretrainning_model = 'bert' #bert,electra
        self.data_path = '/home/none404/hm/Tencent_ads/data/'
        self.train_type = 'ad'  # 'all':全部特征 'ad':ad_id+others 'creative_id+others'
        self.finetune_data = '/home/none404/hm/Tencent_ads/finetune_xuan/finetune_data/'
        self.data_processed = self.finetune_data+'{}_data/'.format(self.train_type) +'processed/'
        self.save_model = '/home/none404/hm/Tencent_ads/finetune_data/' + '{}_model/'.format(self.train_type)
        self.result = '/home/none404/hm/Tencent_ads/finetune_xuan/result/'
        if self.pretrainning_model=='bert':
            self.bert_file = "/home/none404/hm/Tencent_ads/finetune_xuan/ad_model/model.ckpt-0"
            self.bert_config_file = "/home/none404/hm/Tencent_ads/finetune_xuan/ad_model/bert_config.json"
            self.vocab_file = "/home/none404/hm/Tencent_ads/finetune_xuan/ad_model/vocab.txt"



        self.use_origin_bert = False # True:使用原生bert, False:使用动态融合bert
        self.is_avg_pool = True # True: 使用平均avg_pool False:使用CLS
        self.joint = True  #True联合学习任务 False:20分类
        self.joint_rate = [0.85 ,0.15]  #联合学习loss权值

        # self.model_type = 'bilstm'
        self.model_type = 'gru'
        # self.model_type = 'only bert output'
        self.lstm_dim = 256
        self.dropout = 0.9
        # self.loss_name = 'focal_loss'
        self.loss_name = 'normal'
        self.gru_num = 256


        self.save_dict ={'11': 0, '12': 1,  '13': 2,  '14': 3,  '15': 4,  '16': 5,  '17': 6,  '18': 7,  '19': 8,  '110': 9,
                         '21': 10,  '22': 11,  '23': 12,  '24': 13,  '25': 14,  '26': 15,  '27': 16,  '28': 17,  '29': 18,  '210': 19
                         } #保存的字典  第一位为性别



