"""
@file   : run_unsup_consert.py
"""
import os
import torch
import time
from model import Model
from config import set_args
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from dataset import SimDataset, SimTestDataset
import torch.nn.functional as F


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
#             output = model.encode(input_ids=input_ids, attention_mask=input_mask)
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

def cal_cos_sim(embedding1, embedding2):
    embedding1_norm = F.normalize(embedding1, p=2, dim=1)
    embedding2_norm = F.normalize(embedding2, p=2, dim=1)
    return torch.mm(embedding1_norm, embedding2_norm.transpose(0, 1))


if __name__ == '__main__':
    args = set_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # 获取到dataset

    train_dataset = SimDataset(args.train_data)
    valid_dataset = SimTestDataset(args.test_data)

    # 生成Batch,发现的问题，放在epoch里读取，第二个epoch的batch_size会自动变
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False)
    total_steps = int(len(train_dataloader) * args.num_train_epochs / args.gradient_accumulation_steps)

    print("总训练步数为:{}".format(total_steps))
    model = Model()
    if torch.cuda.is_available():
        model.cuda()

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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    loss_fct = torch.nn.CrossEntropyLoss()
    for epoch in range(args.num_train_epochs):
        model.train()
        temp_loss = 0
        count = 0
        for step, batch in enumerate(train_dataloader):
            count += 1
            start_time = time.time()
            input_ids = batch["input_ids"]  # torch.Size([6, 22])
            attention_mask = batch["attention_mask"]

            if torch.cuda.is_available():
                input_ids = input_ids.long().cuda()
                attention_mask = attention_mask.long().cuda()

            s1_embedding, s2_embedding = model(input_ids1=input_ids, attention_mask1=attention_mask)

            cos_sim = cal_cos_sim(s1_embedding, s2_embedding) / args.temperature

            batch_size = cos_sim.size(0)
            assert cos_sim.size() == (batch_size, batch_size)
            labels = torch.arange(batch_size).cuda()
            loss = loss_fct(cos_sim.cuda(), labels)

            temp_loss += loss
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

        # corr, pears = evaluate()
        # s = 'Epoch:{} | cur_epoch_average_loss:{:10f} | spearmanr: {:10f} | pearsonr: {:10f}'.format(epoch, train_loss, corr, pears)
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
