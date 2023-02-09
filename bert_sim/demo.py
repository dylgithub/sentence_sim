# -*- coding: utf-8 -*-
import torch
from transformers import BertTokenizer
import numpy as np
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 返回cuda表示成功
# 或者
# print(torch.cuda.is_available())
text = ["测试数据","测试数据2"]
tokenizer = BertTokenizer.from_pretrained("rbt3")
text1 = "测试数据1"
text2 = "测试数据2"
# token = tokenizer.encode_plus(
#             text=text,
#             text_pair=None,
#             add_special_tokens=True,
#             return_token_type_ids=True
#         )
token = tokenizer(text1, text2, add_special_tokens=True, padding='max_length', truncation=True, max_length=20)
# token = tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=20)
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
# def forward(output1, output2, label, batch_size):
#     margin = 1
#     # print(label)
#     # 欧式距离
#     euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True, p=2)
#     print(torch.add(torch.norm(output1, keepdim=True, dim=1), torch.norm(output2, keepdim=True, dim=1)))
#     euclidean_distance = torch.div(euclidean_distance, torch.add(torch.norm(output1, keepdim=True, dim=1),
#                                                                  torch.norm(output2, keepdim=True, dim=1)))
#     euclidean_distance = torch.reshape(euclidean_distance, [-1])
#     print(euclidean_distance)
#     # pos = label * euclidean_distance
#     # neg = (1 - label) * torch.clamp(margin - euclidean_distance, min=0.0)
#     # print(neg)
#     # return torch.mean(pos + neg) / batch_size / 2
#     # print(1 - label)
#     # loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
#     #                               (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0),
#     #                                                   2)) / batch_size / 2
#     # # label 为1时欧式距离越大，越不相似，loss对应越大
#     # loss_contrastive2 = torch.mean(label * torch.pow(euclidean_distance, 2) +
#     #                                (1 - label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0),
#     #                                                        2)) / batch_size / 2
#
#     # return loss_contrastive, loss_contrastive2
#     # return euclidean_distance

#
# if __name__ == '__main__':
#     output1 = np.random.uniform(0, 1, 1000)
#     output1 = output1.reshape(10, 100)
#     output1 = torch.Tensor(output1)
#     output2 = np.random.uniform(0, 1, 1000)
#     output2 = output2.reshape(10, 100)
#     output2 = torch.Tensor(output2)
#     label = np.array([1., 1., 1., 0., 0., 0., 1., 0., 1., 1.])
#     label = torch.Tensor(label)
#     forward(output1, output2, label, 10)
#     # euclidean_distance = forward(output1, output2, label, 10)
#     # print(euclidean_distance)
#     # loss_contrastive, loss_contrastive2 = forward(output1, output2, label, 10)
#     # print(loss_contrastive, loss_contrastive2)
