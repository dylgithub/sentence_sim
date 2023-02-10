# coding: utf-8
# @File: dataset.py
# @Description:

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer
from tqdm import tqdm
import random


class SimDataset(Dataset):
    def __init__(self, filename):
        # 数据集初始化
        self.tokenizer = BertTokenizer.from_pretrained('F:/pytorch_workplace/sentence_sim/bert_sim/rbt3')
        self.input_ids_text1 = []
        self.token_type_ids_text1 = []
        self.attention_mask_text1 = []
        self.load_data(filename)

    def load_data_bak(self, filename):
        # 加载数据
        print('loading data from:', filename)
        with open(filename, 'r', encoding='utf-8') as rf:
            lines = rf.readlines()
        for line in tqdm(lines, ncols=100):
            text1, text2, label = line.strip().split('\t')
            label_id = int(label)
            token1 = self.tokenizer(text1, add_special_tokens=True, padding='max_length', truncation=True,
                                    max_length=50)
            token2 = self.tokenizer(text2, add_special_tokens=True, padding='max_length', truncation=True,
                                    max_length=50)
            self.input_ids_text1.append(np.array(token1['input_ids']))
            self.token_type_ids_text1.append(np.array(token1['token_type_ids']))
            self.attention_mask_text1.append(np.array(token1['attention_mask']))
            self.input_ids_text2.append(np.array(token2['input_ids']))
            self.token_type_ids_text2.append(np.array(token2['token_type_ids']))
            self.attention_mask_text2.append(np.array(token2['attention_mask']))
            self.label_id.append(label_id)

    def load_data(self, filename):
        # 加载数据
        print('loading data from:', filename)
        df = pd.read_excel(filename)
        query1_list = df["query1"].to_list()
        query2_list = df["query2"].to_list()
        data_list = query1_list + query2_list
        random.shuffle(data_list)
        data_list = data_list[:10000]
        for index in tqdm(range(len(data_list))):
            text = data_list[index]
            token1 = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True,
                                    max_length=50)
            self.input_ids_text1.append(np.array(token1['input_ids']))
            self.token_type_ids_text1.append(np.array(token1['token_type_ids']))
            self.attention_mask_text1.append(np.array(token1['attention_mask']))

    def __getitem__(self, index):
        return {"input_ids": self.input_ids_text1[index], "token_type_ids": self.token_type_ids_text1[index],
                "attention_mask": self.attention_mask_text1[index]}

    def __len__(self):
        return len(self.input_ids_text1)
