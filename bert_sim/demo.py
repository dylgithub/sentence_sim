# -*- coding: utf-8 -*-
import torch
from transformers import BertTokenizer
import numpy as np

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 返回cuda表示成功
# 或者
# print(torch.cuda.is_available())
# text = ["测试数据","测试数据2"]
tokenizer = BertTokenizer.from_pretrained("rbt3")
text1 = "测试数据1"
text2 = "测试数据2"
token = tokenizer(text1, text2, add_special_tokens=True, padding='max_length', truncation=True, max_length=20)
print(token['input_ids'])
print(token['token_type_ids'])
print(token['attention_mask'])
# print(token2char_span_mapping)
# data = [[[1, 0, 0], [1, 0, 0]],
#         [[1, 1, 0], [1, 1, 0]]]
# data_array = np.array(data)
# print(data_array.shape)
# for ent_type_id, token_start_index, token_end_index in zip(*np.where(data_array > 0)):
#     print(ent_type_id, token_start_index, token_end_index)
