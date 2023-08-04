"""
@file   : esimcse.py
"""
import torch
import random
from torch import nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel


class Model(nn.Module):
    def __init__(self, q_size=128, dup_rate=0.32, temperature=0.05, gamma=0.99):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained("F:/pytorch_workplace/sentence_sim/bert_sim/rbt3")
        self.dup_rate = dup_rate  # 数据增广的比例

    def word_repetition(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.size()
        input_ids, attention_mask = input_ids.cpu().tolist(), attention_mask.cpu().tolist()
        repetitied_input_ids, repetitied_attention_mask = [], []
        rep_seq_len = seq_len
        for batch_id in range(batch_size):
            # 一个一个序列进行处理
            sample_mask = attention_mask[batch_id]
            actual_len = sum(sample_mask)  # 计算当前序列的真实长度

            cur_input_ids = input_ids[batch_id]
            # 随机选取dup_len个token
            dup_len = random.randint(a=0, b=max(2, int(self.dup_rate * actual_len)))  # dup_rate越大  可能重复的token越多 否则越少
            dup_word_index = random.sample(list(range(1, actual_len)), k=dup_len)  # 采样出dup_len个token  然后下面进行重复
            r_input_id, r_attention_mask = [], []
            for index, word_id in enumerate(cur_input_ids):
                if index in dup_word_index:
                    r_input_id.append(word_id)
                    r_attention_mask.append(sample_mask[index])
                r_input_id.append(word_id)
                r_attention_mask.append(sample_mask[index])

            after_dup_len = len(r_input_id)
            repetitied_input_ids.append(r_input_id)
            repetitied_attention_mask.append(r_attention_mask)

            assert after_dup_len == dup_len + seq_len
            if after_dup_len > rep_seq_len:
                rep_seq_len = after_dup_len

        for i in range(batch_size):
            after_dup_len = len(repetitied_input_ids[i])
            pad_len = rep_seq_len - after_dup_len
            repetitied_input_ids[i] += [0] * pad_len
            repetitied_attention_mask[i] += [0] * pad_len

        repetitied_input_ids = torch.tensor(repetitied_input_ids, dtype=torch.long).cuda()
        repetitied_attention_mask = torch.tensor(repetitied_attention_mask, dtype=torch.long).cuda()
        return repetitied_input_ids, repetitied_attention_mask

    def forward(self, input_ids, attention_mask, token_type):
        # 这里直接取CLS向量 也可用其他的方式
        bert_output_text1 = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                      token_type_ids=token_type)
        s1_embedding = bert_output_text1[1]

        # 给当前输入的样本拷贝一个正样本
        input_ids2, attention_mask2 = torch.clone(input_ids), torch.clone(attention_mask)
        input_ids2, attention_mask2 = self.word_repetition(input_ids2, attention_mask2)  # 数据增广 重复某些字
        bert_output_text2 = self.bert(input_ids2, attention_mask2)
        s2_embedding = bert_output_text2[1]
        return s1_embedding, s2_embedding, self.bert
