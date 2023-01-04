# coding: utf-8
# @File: train.py
# @Time: 2020/10/10 17:14:07
# @Description:

import os
import torch
import torch.nn as nn
from transformers import BertConfig
from torch.utils.data import DataLoader
from siamese_network_with_cross_attention import BertClassifier
from siam_net_dataset import CNewsDataset
from tqdm import tqdm
from siamese_network_with_cross_attention_config import Config
import torch.nn.functional as F

def main():
    # 参数设置
    config = Config()
    batch_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 10
    # 目前尝试，用对比损失函数，学习率不易设置的过大
    learning_rate = 5e-5  # Learning Rate不宜太大
    # learning_rate = 5e-4  # Learning Rate不宜太大
    # learning_rate = 0.001  # Learning Rate不宜太大

    # 获取到dataset
    train_dataset = CNewsDataset('senteval_cn/BQ/BQ.train.data')
    # train_dataset = CNewsDataset('senteval_cn/BQ/BQ.valid.data')
    # valid_dataset = CNewsDataset('senteval_cn/BQ/BQ.valid.data')
    # test_dataset = CNewsDataset('THUCNews/data/test.txt')

    # 生成Batch,发现的问题，放在epoch里读取，第二个epoch的batch_size会自动变
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = BertClassifier(config).to(device)

    # 优化器
    # optimizer = AdamW(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(1, epochs + 1):
        losses = 0  # 损失
        accuracy = 0  # 准确率

        model.train()

        print("len train_dataloader", len(train_dataloader))
        train_bar = tqdm(train_dataloader, ncols=100)
        for input_ids_text1, token_type_ids_text1, attention_mask_text1, input_ids_text2, token_type_ids_text2, attention_mask_text2, label_id in train_bar:
            # 梯度清零
            model.zero_grad()
            train_bar.set_description('Epoch %i train' % epoch)
            label_id = label_id.float().to(device)
            batch_size = len(label_id)
            # 传入数据，调用model.forward()
            # 注意这里的数据类型转换
            output = model(
                input_ids_text1=input_ids_text1.long().to(device),
                attention_mask_text1=attention_mask_text1.long().to(device),
                token_type_ids_text1=token_type_ids_text1.long().to(device),
                input_ids_text2=input_ids_text2.long().to(device),
                attention_mask_text2=attention_mask_text2.long().to(device),
                token_type_ids_text2=token_type_ids_text2.long().to(device),
            )
            # 计算loss
            # sim_score, loss = criterion(cls_text1, cls_text2, label_id, batch_size)
            loss = criterion(output, label_id.to(device))
            losses += loss.item()
            pred_labels = torch.argmax(output, dim=1)  # 预测出的label
            acc = torch.sum(pred_labels == label_id.to(device)).item() / len(pred_labels)  # acc
            accuracy += acc

            loss.backward()
            optimizer.step()
            train_bar.set_postfix(loss=loss.item(), acc=acc)
            # train_bar.set_postfix(loss=loss.item())

        average_loss = losses / len(train_dataloader)
        average_acc = accuracy / len(train_dataloader)
        print('\tTrain ACC:', average_acc, '\tLoss:', average_loss)
        #
        # # 验证
        # model.eval()
        # losses = 0  # 损失
        # accuracy = 0  # 准确率
        # valid_bar = tqdm(valid_dataloader, ncols=100)
        # for input_ids, token_type_ids, attention_mask, label_id in valid_bar:
        #     valid_bar.set_description('Epoch %i valid' % epoch)
        #
        #     output = model(
        #         input_ids=input_ids.long().to(device),
        #         attention_mask=attention_mask.long().to(device),
        #         token_type_ids=token_type_ids.long().to(device),
        #     )
        #
        #     loss = criterion(output, label_id.to(device))
        #     losses += loss.item()
        #
        #     pred_labels = torch.argmax(output, dim=1)  # 预测出的label
        #     acc = torch.sum(pred_labels == label_id.to(device)).item() / len(pred_labels)  # acc
        #     accuracy += acc
        #     valid_bar.set_postfix(loss=loss.item(), acc=acc)
        #
        # average_loss = losses / len(valid_dataloader)
        # average_acc = accuracy / len(valid_dataloader)
        #
        # print('\tValid ACC:', average_acc, '\tLoss:', average_loss)
        #
        # if not os.path.exists('models'):
        #     os.makedirs('models')
        #
        # # 判断并保存验证集上表现最好的模型
        # if average_acc > best_acc:
        #     best_acc = average_acc
        #     torch.save(model.state_dict(), 'models/best_model.pkl')


if __name__ == '__main__':
    main()
