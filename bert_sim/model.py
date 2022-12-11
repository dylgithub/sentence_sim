# coding: utf-8
# @File: model.py
# @Description:

import torch
import torch.nn as nn
from transformers import BertModel


# Bert
class BertClassifier(nn.Module):
    def __init__(self, bert_config, num_labels):
        super().__init__()
        # 定义BERT模型
        self.bert = BertModel.from_pretrained("rbt3")
        # 定义分类器
        self.classifier = nn.Linear(bert_config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # BERT的输出
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 取[CLS]位置的pooled output
        pooled = bert_output[1]
        # 分类
        logits = self.classifier(pooled)
        # 直接返回，不要softmax，因为算Loss时CrossEntropyLoss里面会进行softmax操作
        return logits

# Bert+BiLSTM，用法与BertClassifier一样，可直接在train里面调用
# class BertLstmClassifier(nn.Module):
#     def __init__(self, bert_config):
#         super().__init__()
#         self.bert = BertModel(config=bert_config)
#         self.lstm = nn.LSTM(input_size=bert_config.hidden_size, hidden_size=bert_config.hidden_size, num_layers=2, batch_first=True, bidirectional=True)
#         self.classifier = nn.Linear(bert_config.hidden_size*2, bert_config.num_labels)  # 双向LSTM 需要乘以2
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, input_ids, attention_mask, token_type_ids):
#         output, pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         out, _ = self.lstm(output)
#         logits = self.classifier(out[:, -1, :]) # 取最后时刻的输出
#         return self.softmax(logits)
