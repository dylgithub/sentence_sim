"""
@file   : run_unsup_simcse.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-08-23
"""
import os
import torch
import time
from torch import nn
from model import Model
from config import set_args
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel, get_linear_schedule_with_warmup
from dataset import SimDataset


def cal_cos_sim(embedding1, embedding2):
    embedding1_norm = F.normalize(embedding1, p=2, dim=1)
    embedding2_norm = F.normalize(embedding2, p=2, dim=1)
    return torch.mm(embedding1_norm, embedding2_norm.transpose(0, 1))  # (batch_size, batch_size)


class Loss(torch.nn.Module):
    """
    Loss function.
    """

    def __init__(self):
        super(Loss, self).__init__()
        self.temperature = 0.05
        self.gamma = 0.99
        self.q = []  # 积攒负样本的队列
        self.q_size = 16  # 队列长度
        self.loss_fct = nn.CrossEntropyLoss()
        # 下面这个是为了获取上一个batch中的样本编码向量
        self.moco_config = BertConfig.from_pretrained('F:/pytorch_workplace/sentence_sim/bert_sim/rbt3/config.json')
        self.moco_config.hidden_dropout_prob = 0.0  # 不用dropout
        self.moco_config.attention_probs_dropout_prob = 0.0  # 不用dropout
        self.moco_bert = BertModel.from_pretrained('F:/pytorch_workplace/sentence_sim/bert_sim/rbt3',
                                                   config=self.moco_config)

    def forward(self, s1_embedding, s2_embedding, input_ids1, attention_mask1, bert):
        # 计算cos
        cos_sim = cal_cos_sim(s1_embedding, s2_embedding) / self.temperature  # (batch_size, batch_size)

        batch_size = cos_sim.size(0)
        assert cos_sim.size() == (batch_size, batch_size)
        negative_samples = None
        if len(self.q) > 0:
            # 从队列中取出负样本
            negative_samples = torch.cat(self.q[:self.q_size], dim=0)  # (q_size, 768)

        if len(self.q) + batch_size >= self.q_size:
            # 这个批次的样本准备加入到负样本队列  测试一下  加入进去 是否超过最大队列长度 如果超过 将队头多余的出队
            del self.q[:batch_size]

        # 将当前batch作为负样本 加入到负样本队列
        with torch.no_grad():
            self.q.append(
                self.moco_bert(input_ids1, attention_mask1)[1])

        labels = torch.arange(batch_size).cuda()
        if negative_samples is not None:
            batch_size += negative_samples.size(0)  # batch_size + 负样本的个数
            cos_sim_with_neg = self.cal_cos_sim(s1_embedding,
                                                negative_samples) / self.temperature  # 当前batch和之前负样本的cos (N, M)
            cos_sim = torch.cat([cos_sim, cos_sim_with_neg], dim=1)  # (N, N+M)

        for encoder_param, moco_encoder_param in zip(bert.parameters(), self.moco_bert.parameters()):
            moco_encoder_param.data = self.gamma * moco_encoder_param.data + (1. - self.gamma) * encoder_param.data

        loss = self.loss_fct(cos_sim, labels)
        return loss


if __name__ == '__main__':
    args = set_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)
    # 获取到dataset
    train_dataset = SimDataset('F:/pytorch_workplace/sentence_sim/bert_sim/other_data/train.xlsx')
    valid_dataset = SimDataset('F:/pytorch_workplace/sentence_sim/bert_sim/other_data/test.xlsx')

    # 生成Batch,发现的问题，放在epoch里读取，第二个epoch的batch_size会自动变
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    total_steps = int(len(train_dataloader) * args.num_train_epochs / args.gradient_accumulation_steps)

    # 初始化模型
    model = Model().to(device)
    # 优化器
    # optimizer = AdamW(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 损失函数
    criterion = Loss()

    # 获取模型所有参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # 设置优化器
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)

    for epoch in range(args.num_train_epochs):
        model.train()
        temp_loss = 0
        count = 0
        for step, batch in enumerate(train_dataloader):
            count += 1
            start_time = time.time()
            input_ids = batch["input_ids"]
            attention_mask_ids = batch["attention_mask"]
            token_type_ids = batch["token_type_ids"]

            s1_embedding, s2_embedding, bert = model(
                input_ids=input_ids.long().to(device),
                attention_mask=attention_mask_ids.long().to(device),
                token_type=token_type_ids.long().to(device),
            )
            loss = criterion(s1_embedding, s2_embedding, input_ids.long(), attention_mask_ids.long(), bert)
            # temp_loss += loss
            ss = 'Epoch:{}, Step:{}, Loss:{:10f}, Time_cost:{:10f}'.format(epoch, step, loss, time.time() - start_time)
            print(ss)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # 损失进行回传
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # 当训练步数整除累积步数时，进行参数优化
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # train_loss = temp_loss / count
        # s = 'Epoch:{} | cur_epoch_average_loss:{:10f} | spearmanr: {:10f} | pearsonr: {:10f}'.format(epoch, train_loss, corr, pears)
        # print(s)
        # corr, pears = evaluate()
        # logs_path = os.path.join(args.output_dir, 'logs.txt')
        # with open(logs_path, 'a+') as f:
        #     s += '\n'
        #     f.write(s)

        # 每个epoch进行完，则保存模型
        output_dir = os.path.join(args.output_dir, "Epoch-{}.bin".format(epoch))
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), output_dir)
        # 清空cuda缓存
        torch.cuda.empty_cache()
