"""
@file   : run_unsup_simcse.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-08-23
"""
import os
import random
import torch
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import nn
from model import Model
from config import set_args
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import SimDataset, SimTestDataset
from einops import repeat, rearrange
from transformers import BertTokenizer, get_linear_schedule_with_warmup, AdamW


# def evaluate():
#     sent1, sent2, label = load_test_data(args.test_data)
#     all_a_vecs = []
#     all_b_vecs = []
#     all_labels = []
#     model.eval()
#     for s1, s2, lab in tqdm(zip(sent1, sent2, label)):
#         input_ids, input_mask, segment_ids = get_sent_id_tensor([s1, s2])
#         lab = torch.tensor([lab], dtype=torch.float)
#         if torch.cuda.is_available():
#             input_ids, input_mask, segment_ids = input_ids.cuda(), input_mask.cuda(), segment_ids.cuda()
#             lab = lab.cuda()
#
#         with torch.no_grad():
#             output = model(input_ids=input_ids, attention_mask=input_mask, encoder_type='cls')
#
#         all_a_vecs.append(output[0].cpu().numpy())
#         all_b_vecs.append(output[1].cpu().numpy())
#         all_labels.extend(lab.cpu().numpy())
#
#     all_a_vecs = np.array(all_a_vecs)
#     all_b_vecs = np.array(all_b_vecs)
#     all_labels = np.array(all_labels)
#
#     a_vecs = l2_normalize(all_a_vecs)
#     b_vecs = l2_normalize(all_b_vecs)
#     sims = (a_vecs * b_vecs).sum(axis=1)
#     corrcoef = compute_corrcoef(all_labels, sims)
#     pearsonr = compute_pearsonr(all_labels, sims)
#     return corrcoef, pearsonr


def compute_loss(y_pred, tao=0.05, device="cuda"):
    idxs = torch.arange(0, y_pred.shape[0], device=device)
    y_true = idxs + 1 - idxs % 2 * 2
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    similarities = similarities - torch.eye(y_pred.shape[0], device=device) * 1e12
    similarities = similarities / tao
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


if __name__ == '__main__':
    args = set_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)
    # 获取到dataset

    train_dataset = SimDataset(args.train_data)
    valid_dataset = SimTestDataset(args.test_data)

    # 生成Batch,发现的问题，放在epoch里读取，第二个epoch的batch_size会自动变
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    total_steps = int(len(train_dataloader) * args.num_train_epochs / args.gradient_accumulation_steps)

    print("总训练步数为:{}".format(total_steps))
    model = Model().to(device)

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
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)

    for epoch in range(args.num_train_epochs):
        model.train()
        temp_loss = 0
        count = 0
        for step, batch in enumerate(train_dataloader):
            count += 1
            start_time = time.time()
            input_ids = batch["input_ids"]  # torch.Size([6, 22])

            input_ids = repeat(input_ids, 'b l -> b new_axis l', new_axis=2)
            input_ids = rearrange(input_ids, 'b x l -> (b x) l')
            #
            attention_mask_ids = batch["attention_mask"]

            attention_mask_ids = repeat(attention_mask_ids, 'b l -> b new_axis l', new_axis=2)
            attention_mask_ids = rearrange(attention_mask_ids, 'b x l -> (b x) l')

            # print(input_ids[:2])    # torch.Size([12, 22])   # 2 * batch_size, max_len
            # print(attention_mask_ids.size())    # torch.Size([12, 22])   # 2 * batch_size, max_len

            # input_ids = batch["input_ids"]
            # attention_mask = batch["attention_mask"]
            token_type_ids = batch["token_type_ids"]
            outputs = model(
                input_ids=input_ids.long().to(device),
                attention_mask=attention_mask_ids.long().to(device),
                # token_type_ids=token_type_ids.long().to(device),
            )
            # outputs = model(input_ids, attention_mask_ids, encoder_type='cls')
            loss = compute_loss(outputs)
            temp_loss += loss.item()

            # 将损失值放到Iter中，方便观察
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
        #
        # corr, pears = evaluate()
        # s = 'Epoch:{} | cur_epoch_average_loss:{:10f} |spearmanr: {:10f} | pearsonr: {:10f}'.format(epoch, train_loss,
        #                                                                                             corr, pears)
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
