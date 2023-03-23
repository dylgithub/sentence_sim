# coding: utf-8

import os
import torch
from torch.utils.data import DataLoader
from siamese_network import BertClassifier
from siam_net_dataset import SimDataset
from tqdm import tqdm
import torch.nn.functional as F


def distance_metric(x, y):
    return 1 - F.cosine_similarity(x, y)


# 自定义ContrastiveLoss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    # sentences-transformer中的实现方法
    def forward(self, output1, output2, labels, batch_size):
        # 余弦相似度的值越大越相似，这个值会越小
        distances = distance_metric(output1, output2)
        # 正样本对，余弦相似度的值越大越相似，LOSS值应该越小
        losses = 0.5 * (
                labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
        return losses.sum()


class OnlineContrastiveLoss(torch.nn.Module):
    """
    OnlineContrastiveLoss function.
    """

    def __init__(self, margin=1.0):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin

    # sentences-transformer中的实现方法
    def forward(self, output1, output2, labels, batch_size):
        distance_matrix = distance_metric(output1, output2)
        negs = distance_matrix[labels == 0]
        poss = distance_matrix[labels == 1]

        # select hard positive and hard negative pairs
        negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

        positive_loss = positive_pairs.pow(2).sum()
        negative_loss = F.relu(self.margin - negative_pairs).pow(2).sum()
        loss = positive_loss + negative_loss
        return loss


def main():
    # 参数设置
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 15
    # 目前尝试，用对比损失函数，学习率不易设置的过大
    learning_rate = 5e-6  # Learning Rate不宜太大
    # learning_rate = 5e-4  # Learning Rate不宜太大
    # learning_rate = 0.001  # Learning Rate不宜太大

    # 获取到dataset
    train_dataset = SimDataset('F:/pytorch_workplace/sentence_sim/bert_sim/other_data/train.xlsx')
    valid_dataset = SimDataset('F:/pytorch_workplace/sentence_sim/bert_sim/other_data/test.xlsx')
    # valid_dataset = CNewsDataset('senteval_cn/BQ/BQ.valid.data')
    # test_dataset = CNewsDataset('THUCNews/data/test.txt')

    # 生成Batch,发现的问题，放在epoch里读取，第二个epoch的batch_size会自动变
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = BertClassifier().to(device)

    # 优化器
    # optimizer = AdamW(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 损失函数
    # criterion = ContrastiveLoss()
    # criterion = OnlineContrastiveLoss()
    criterion = torch.nn.CrossEntropyLoss()

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
            label_id = label_id.to(device)
            # 传入数据，调用model.forward()
            # 注意这里的数据类型转换
            _, _, output = model(
                input_ids_text1=input_ids_text1.long().to(device),
                attention_mask_text1=attention_mask_text1.long().to(device),
                token_type_ids_text1=token_type_ids_text1.long().to(device),
                input_ids_text2=input_ids_text2.long().to(device),
                attention_mask_text2=attention_mask_text2.long().to(device),
                token_type_ids_text2=token_type_ids_text2.long().to(device),
            )
            # 计算loss
            loss = criterion(output, label_id)
            losses += loss.item()
            pred_labels = torch.argmax(output, dim=1)  # 预测出的label
            acc = torch.sum(pred_labels == label_id).item() / len(pred_labels)  # acc
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
        model.eval()
        losses = 0  # 损失
        accuracy = 0  # 准确率
        valid_bar = tqdm(valid_dataloader, ncols=100)
        for input_ids_text1, token_type_ids_text1, attention_mask_text1, input_ids_text2, token_type_ids_text2, attention_mask_text2, label_id in valid_bar:
            valid_bar.set_description('Epoch %i valid' % epoch)
            label_id = label_id.to(device)
            _, _, output = model(
                input_ids_text1=input_ids_text1.long().to(device),
                attention_mask_text1=attention_mask_text1.long().to(device),
                token_type_ids_text1=token_type_ids_text1.long().to(device),
                input_ids_text2=input_ids_text2.long().to(device),
                attention_mask_text2=attention_mask_text2.long().to(device),
                token_type_ids_text2=token_type_ids_text2.long().to(device),
            )

            # 计算loss
            loss = criterion(output, label_id)
            losses += loss.item()
            pred_labels = torch.argmax(output, dim=1)  # 预测出的label
            acc = torch.sum(pred_labels == label_id).item() / len(pred_labels)  # acc
            accuracy += acc
            valid_bar.set_postfix(loss=loss.item(), acc=acc)

        average_loss = losses / len(valid_dataloader)
        average_acc = accuracy / len(valid_dataloader)

        print('\tValid ACC:', average_acc, '\tLoss:', average_loss)

        if not os.path.exists('models'):
            os.makedirs('models')

        # 判断并保存验证集上表现最好的模型
        if average_acc > best_acc:
            best_acc = average_acc
            # torch.save(model.state_dict(), 'models/best_model.pkl')

    # torch.save(model.state_dict(), 'models/siam_net/best_model.pkl')


if __name__ == '__main__':
    # 最好0.7
    main()
