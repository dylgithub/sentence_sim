import argparse


def set_args():
    parser = argparse.ArgumentParser('--PromptBert')
    parser.add_argument('--train_data_path', default='F:/pytorch_workplace/sentence_sim/bert_sim/other_data/train.xlsx', type=str, help='训练数据集')
    parser.add_argument('--dev_data_path', default='F:/pytorch_workplace/sentence_sim/bert_sim/other_data/test.xlsx', type=str, help='测试数据集')
    parser.add_argument('--max_len', default=64, type=int, help='句子的最大长度')
    parser.add_argument('--train_batch_size', default=16, type=int, help='训练批次的大小')
    parser.add_argument('--dev_batch_size', default=16, type=int, help='训练批次的大小')
    parser.add_argument('--num_train_epochs', default=3, type=int, help='训练几轮')
    parser.add_argument('--learning_rate', default=9e-6, type=float, help='学习率大小')
    parser.add_argument('--bert_pretrain_path', default='F:/pytorch_workplace/sentence_sim/bert_sim/rbt3', type=str, help='预训练模型路径')
    parser.add_argument('--output_dir', default='./output', type=str, help='模型输出目录')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='梯度积聚的大小')
    return parser.parse_args()
