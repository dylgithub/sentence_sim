import os
import torch
import numpy as np
from tqdm import tqdm
from config import set_args
from transformers.models.bert import BertTokenizer
from torch.utils.data import DataLoader
from model import PromptBERT
from utils import l2_normalize, compute_corrcoef, compute_pearsonr
from transformers import AdamW, get_linear_schedule_with_warmup
from data_helper import load_data, SentDataSet, collate_func, convert_token_id


if __name__ == '__main__':
    args = set_args()

    # os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain_path)
    # 加入一个特殊token: [X]
    tokenizer.add_special_tokens({'additional_special_tokens': ['[X]']})
    mask_id = tokenizer.mask_token_id

    train_df = load_data(args.train_data_path, tokenizer)
    print(train_df)
    # train_dataset = SentDataSet(train_df, tokenizer)
    # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size,
    #                               collate_fn=collate_func)
    #
    # num_train_steps = int(
    #     len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    #
    # model = PromptBERT(mask_id=mask_id)
    #
    # if torch.cuda.is_available():
    #     model.cuda()
    #
    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    #
    # warmup_steps = 0.05 * num_train_steps
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)
    # model.bert.resize_token_embeddings(len(tokenizer))
    #
    # for epoch in range(args.num_train_epochs):
    #     model.train()
    #     for step, batch in enumerate(train_dataloader):
    #         if torch.cuda.is_available():
    #             batch = (t.cuda() for t in batch)
    #         (sent_prompt1_input_ids, sent_prompt1_attention_mask, sent_prompt1_token_type_ids,
    #          sent_template1_input_ids, sent_template1_attention_mask, sent_template1_token_type_ids,
    #          sent_prompt2_input_ids, sent_prompt2_attention_mask, sent_prompt2_token_type_ids,
    #          sent_template2_input_ids, sent_template2_attention_mask, sent_template2_token_type_ids) = batch
    #
    #         prompt_embedding0 = model(prompt_input_ids=sent_prompt1_input_ids,
    #                                   prompt_attention_mask=sent_prompt1_attention_mask,
    #                                   prompt_token_type_ids=sent_prompt1_token_type_ids,
    #                                   template_input_ids=sent_template1_input_ids,
    #                                   template_attention_mask=sent_template1_attention_mask,
    #                                   template_token_type_ids=sent_template1_token_type_ids)
    #         prompt_embedding1 = model(prompt_input_ids=sent_prompt2_input_ids,
    #                                   prompt_attention_mask=sent_prompt2_attention_mask,
    #                                   prompt_token_type_ids=sent_prompt2_token_type_ids,
    #                                   template_input_ids=sent_template2_input_ids,
    #                                   template_attention_mask=sent_template2_attention_mask,
    #                                   template_token_type_ids=sent_template2_token_type_ids)
    #
    #         loss = calc_loss(prompt_embedding0, prompt_embedding1)
    #         if args.gradient_accumulation_steps > 1:
    #             loss = loss / args.gradient_accumulation_steps
    #
    #         print('Epoch:{}, Step:{}, Loss:{:10f}'.format(epoch, step, loss))
    #
    #         loss.backward()
    #         # nn.utils.clip_grad_norm(model.parameters(), max_norm=20, norm_type=2)   # 是否进行梯度裁剪
    #         if (step + 1) % args.gradient_accumulation_steps == 0:
    #             optimizer.step()
    #             scheduler.step()
    #             optimizer.zero_grad()
    #
    #     corrcoef, pearsonr = evaluate()
    #     ss = 'epoch:{}, spearmanr:{:10f}, pearsonr:{:10f}'.format(epoch, corrcoef, pearsonr)
    #     log_path = os.path.join(args.output_dir, 'logs.txt')
    #     with open(log_path, 'a+', encoding='utf8') as f:
    #         ss += '\n'
    #         f.write(ss)
    #
    #     model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    #     output_model_file = os.path.join(args.output_dir, "epoch{}_ckpt.bin".format(epoch))
    #     torch.save(model_to_save.state_dict(), output_model_file)


