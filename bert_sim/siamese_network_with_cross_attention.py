# coding: utf-8
# @File: model.py
# @Description:

import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F


# 参数过多，对于简单的任务容易过拟合
class BertClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义BERT模型
        self.bert = BertModel.from_pretrained("rbt3")
        # 定义分类器
        self.classifier = nn.Linear(config.lstm_hidden_size * 4, config.num_class)
        self.lstm = nn.LSTM(768 * 4, config.lstm_hidden_size, batch_first=True, bidirectional=False)

    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)

    def apply_multiple(self, x):
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        return torch.cat([p1, p2], -1)

    def cross_attention(self, x1, x2):
        attention = torch.matmul(x1, x2.transpose(1, 2))
        weight1 = F.softmax(attention, dim=-1)
        x1_align = torch.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(1, 2), dim=-1)
        x2_align = torch.matmul(weight2, x1)
        return x1_align, x2_align

    # 同构孪生网络
    def forward(self, input_ids_text1, token_type_ids_text1, attention_mask_text1, input_ids_text2,
                token_type_ids_text2, attention_mask_text2):
        # BERT的输出
        bert_output_text1 = self.bert(input_ids=input_ids_text1, attention_mask=attention_mask_text1,
                                      token_type_ids=token_type_ids_text1)
        bert_output_text2 = self.bert(input_ids=input_ids_text2, attention_mask=attention_mask_text2,
                                      token_type_ids=token_type_ids_text2)
        # [batch_size, seq_length, 768]
        last_hidden_state_text1 = bert_output_text1[0]
        last_hidden_state_text2 = bert_output_text2[0]
        q1_align, q2_align = self.cross_attention(last_hidden_state_text1, last_hidden_state_text2)
        # [batch_size, seq_length, 768 * 4]
        q1_combined = torch.cat([last_hidden_state_text1, q1_align, self.submul(last_hidden_state_text1, q1_align)], -1)
        q2_combined = torch.cat([last_hidden_state_text2, q2_align, self.submul(last_hidden_state_text2, q2_align)], -1)
        # [batch_size, seq_length, lstm_hidden_size]
        q1_compose, _ = self.lstm(q1_combined)
        q2_compose, _ = self.lstm(q2_combined)
        q1_req = self.apply_multiple(q1_compose)
        q2_req = self.apply_multiple(q2_compose)
        x = torch.cat([q1_req, q2_req], -1)
        logit = self.classifier(x)
        return logit
