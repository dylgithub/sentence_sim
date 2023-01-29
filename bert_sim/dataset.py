# coding: utf-8
# @File: dataset.py
# @Description:

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer
from tqdm import tqdm


class CNewsDataset(Dataset):
    def __init__(self, filename):
        # 数据集初始化
        self.tokenizer = BertTokenizer.from_pretrained('rbt3')
        self.input_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.label_id = []
        self.load_data(filename)

    def load_data(self, filename):
        # 加载数据
        print('loading data from:', filename)
        df = pd.read_excel(filename)
        label_list = df["label"].to_list()
        query1_list = df["query1"].to_list()
        query2_list = df["query2"].to_list()
        for index in tqdm(range(len(label_list))):
            text1, text2, label = query1_list[index], query2_list[index], label_list[index]
            label_id = int(label)
            token = self.tokenizer(text1, text2, add_special_tokens=True, padding='max_length', truncation=True,
                                   max_length=200)
            self.input_ids.append(np.array(token['input_ids']))
            self.token_type_ids.append(np.array(token['token_type_ids']))
            self.attention_mask.append(np.array(token['attention_mask']))
            self.label_id.append(label_id)

    def __getitem__(self, index):
        return self.input_ids[index], self.token_type_ids[index], self.attention_mask[index], self.label_id[index]

    def __len__(self):
        return len(self.input_ids)


if __name__ == '__main__':
    cd = CNewsDataset("THUCNews/data/train.txt")
