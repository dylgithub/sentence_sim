# coding: utf-8
# @File: train.py
# @Time: 2020/10/10 17:14:07
# @Description:

import os
import torch
import torch.nn as nn
from transformers import BertConfig, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from cosert_model import BertClassifier
from cosert_dataset import TrainDataset, ValidDataset
from tqdm import tqdm
import torch.nn.functional as F



# （a,b,label）会被处理为[a,b],[label,label]
# 所以奇偶之间是文本对，label进行间隔取就行
def calc_loss(y_true, y_pred):
    # 1. 取出真实的标签
    y_true = y_true[::2]  # tensor([1, 0, 1]) 真实的标签

    # 2. 对输出的句子向量进行l2归一化   后面只需要对应为相乘  就可以得到cos值了
    norms = (y_pred ** 2).sum(axis=1, keepdims=True) ** 0.5
    # y_pred = y_pred / torch.clip(norms, 1e-8, torch.inf)
    y_pred = y_pred / norms

    # 3. 奇偶向量相乘
    y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * 20

    # 4. 取出负例-正例的差值
    y_pred = y_pred[:, None] - y_pred[None, :]  # 这里是算出所有位置 两两之间余弦的差值
    # 矩阵中的第i行j列  表示的是第i个余弦值-第j个余弦值
    y_true = y_true[:, None] < y_true[None, :]  # 取出负例-正例的差值
    y_true = y_true.float()
    y_pred = y_pred - (1 - y_true) * 1e12
    y_pred = y_pred.view(-1)
    if torch.cuda.is_available():
        y_pred = torch.cat((torch.tensor([0]).float().cuda(), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
    else:
        y_pred = torch.cat((torch.tensor([0]).float(), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1

    return torch.logsumexp(y_pred, dim=0)

def main():
    # 参数设置
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 15
    learning_rate = 2e-5  # Learning Rate不宜太大

    # 获取到dataset
    train_dataset = TrainDataset('F:/pytorch_workplace/sentence_sim/bert_sim/other_data/train.xlsx')
    valid_dataset = ValidDataset('F:/pytorch_workplace/sentence_sim/bert_sim/other_data/test.xlsx')
    # test_dataset = CNewsDataset('THUCNews/data/test.txt')

    # 生成Batch, 注意这里train的DataLoader中shuffle不可设置为True
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 读取BERT的配置文件
    bert_config = BertConfig.from_pretrained('/bert_sim/rbt3')
    num_labels = 2

    # 初始化模型
    model = BertClassifier(bert_config, num_labels).to(device)

    # 优化器
    # optimizer = AdamW(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 学习率预热
    # 在预热期间，学习率从0线性增加到优化器中的初始lr
    # 在预热阶段之后创建一个schedule，使其学习率从优化器中的初始lr线性降低到0
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0.05 * total_steps,
                                                num_training_steps=total_steps)

    best_acc = 0

    for epoch in range(1, epochs + 1):
        losses = 0  # 损失
        accuracy = 0  # 准确率

        model.train()
        train_bar = tqdm(train_dataloader, ncols=100)
        for input_ids, token_type_ids, attention_mask, label_id in train_bar:
            # 梯度清零
            model.zero_grad()
            train_bar.set_description('Epoch %i train' % epoch)

            # 传入数据，调用model.forward()
            # 注意这里的数据类型转换
            output = model(
                input_ids=input_ids.long().to(device),
                attention_mask=attention_mask.long().to(device),
                token_type_ids=token_type_ids.long().to(device),
            )

            # 计算loss
            loss = calc_loss(label_id.to(device), output)
            losses += loss.item()

            # pred_labels = torch.argmax(output, dim=1)  # 预测出的label
            # acc = torch.sum(pred_labels == label_id.to(device)).item() / len(pred_labels)  # acc
            # accuracy += acc

            loss.backward()
            optimizer.step()
            scheduler.step()
            # train_bar.set_postfix(loss=loss.item(), acc=acc)
            train_bar.set_postfix(loss=loss.item())

        average_loss = losses / len(train_dataloader)
        # average_acc = accuracy / len(train_dataloader)

        # print('\tTrain ACC:', average_acc, '\tLoss:', average_loss)
        print('\tLoss:', average_loss)

        #验证
        model.eval()
        accuracy = 0  # 准确率
        valid_bar = tqdm(valid_dataloader, ncols=100)
        for input_ids_text1, token_type_ids_text1, attention_mask_text1, input_ids_text2, token_type_ids_text2, attention_mask_text2, label_id in valid_bar:
            valid_bar.set_description('Epoch %i valid' % epoch)
            label_id = label_id.float().to(device)
            text1_cls = model(
                input_ids=input_ids_text1.long().to(device),
                attention_mask=attention_mask_text1.long().to(device),
                token_type_ids=token_type_ids_text1.long().to(device),
            )

            text2_cls = model(
                input_ids=input_ids_text2.long().to(device),
                attention_mask=attention_mask_text2.long().to(device),
                token_type_ids=token_type_ids_text2.long().to(device),
            )
            sim_score = F.cosine_similarity(text1_cls, text2_cls)

            pred_labels = (sim_score > 0.5).float()
            # print(pred_labels)
            acc = torch.sum(torch.eq(pred_labels, label_id)).item() / len(label_id)  # acc
            accuracy += acc
            valid_bar.set_postfix(acc=acc)

        average_acc = accuracy / len(valid_dataloader)

        print('\tValid ACC:', average_acc)
        #
        # if not os.path.exists('models'):
        #     os.makedirs('models')
        #
        # # 判断并保存验证集上表现最好的模型
        # if average_acc > best_acc:
        #     best_acc = average_acc
        #     torch.save(model.state_dict(), 'models/best_model.pkl')


if __name__ == '__main__':
    # 最优 Valid ACC: 0.879
    main()
