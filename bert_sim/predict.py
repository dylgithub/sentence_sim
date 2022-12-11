# coding: utf-8
# @File: predict.py
# @Description:

import torch
from model import BertClassifier
from transformers import BertTokenizer, BertConfig


labels = ['房产', '财经', '教育', '科技', '时政', '体育', '游戏', '娱乐']
bert_config = BertConfig.from_pretrained('rbt3')

# 定义模型
model = BertClassifier(bert_config, len(labels))

# 加载训练好的模型
model.load_state_dict(torch.load('models/best_model.pkl', map_location=torch.device('cpu')))
model.eval()

tokenizer = BertTokenizer.from_pretrained('rbt3')

print('新闻类别分类')
while True:
    text = input('Input: ')
    token = tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=512)
    input_ids = token['input_ids']
    attention_mask = token['attention_mask']
    token_type_ids = token['token_type_ids']

    input_ids = torch.tensor([input_ids], dtype=torch.long)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long)
    token_type_ids = torch.tensor([token_type_ids], dtype=torch.long)

    predicted = model(
        input_ids,
        attention_mask,
        token_type_ids,
    )
    pred_label = torch.argmax(predicted, dim=1)

    print('Label:', labels[pred_label])
