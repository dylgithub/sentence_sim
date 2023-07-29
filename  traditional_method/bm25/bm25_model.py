from gensim.summarization import bm25
import jieba
import jieba.posseg as pseg


class BM25Model:
    def __init__(self, data_list):
        self.data_list = data_list
        self.corpus = self.load_corpus()

    def bm25_similarity(self, query, num_best=1):
        query = jieba.lcut(query)  # 分词
        bm = bm25.BM25(self.corpus)
        # average_idf = sum(map(lambda k: float(bm.idf[k]), bm.idf.keys())) / len(bm.idf.keys())
        scores = bm.get_scores(query)
        print(scores)
        id_score = [(i, score) for i, score in enumerate(scores)]
        id_score.sort(key=lambda e: e[1], reverse=True)
        return id_score[0: num_best]

    def load_corpus(self):
        corpus = [jieba.lcut(data) for data in self.data_list]
        return corpus


if __name__ == '__main__':
    data_list = ["小丁的文章不好看", "我讨厌小丁的创作内容", "我非常喜欢小丁写的文章"]
    BM25 = BM25Model(data_list)
    query = "我喜欢小丁写的文章"
    print(BM25.bm25_similarity(query, 1))
