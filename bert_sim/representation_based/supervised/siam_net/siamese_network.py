# coding: utf-8
# @File: model.py
# @Description:

import torch
import torch.nn as nn
from transformers import BertModel


# Bert
class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义BERT模型
        self.bert = BertModel.from_pretrained("F:/pytorch_workplace/sentence_sim/bert_sim/rbt3")

    # 同构孪生网络
    def forward(self, input_ids_text1, token_type_ids_text1, attention_mask_text1, input_ids_text2, token_type_ids_text2, attention_mask_text2):
        # BERT的输出
        bert_output_text1 = self.bert(input_ids=input_ids_text1, attention_mask=attention_mask_text1, token_type_ids=token_type_ids_text1)
        bert_output_text2 = self.bert(input_ids=input_ids_text2, attention_mask=attention_mask_text2, token_type_ids=token_type_ids_text2)
        # 取[CLS]位置的pooled output
        cls_text1 = bert_output_text1[1]
        cls_text2 = bert_output_text2[1]
        return cls_text1, cls_text2

