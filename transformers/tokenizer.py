from tqdm import tqdm
from nltk.tokenize import TreebankWordDetokenizer, TreebankWordTokenizer


class BaseTokenizer:
    """
    基于 jieba 的分词器，用于分词、编码和词表管理。
    """

    unk_token = "<unk>"
    pad_token = "<pad>"  # 填充
    sos_token = "<sos>"  # 开始
    eos_token = "<eos>"  # 结束

    @classmethod
    def tokenize(cls, sentence) -> list[str]:
        """
        使用 jieba 对输入句子进行分词。

        参数:
            sentence (str): 输入的句子。
        """
        pass

    @classmethod
    def build_vocab(cls, sentences, vocab_file):
        """
        构建词表并保存到文件。

        :param sentences: 句子列表。
        :param vocab_file: 保存词表的文件路径。
        """
        unique_words = set()
        for sentence in tqdm(sentences, desc="分词"):
            # 收集所有唯一词
            for word in cls.tokenize(sentence):
                if word.strip() != "":  # 忽略空字符串
                    unique_words.add(word)

        # 将 <unk> 放在词表首位
        vocab_list = [
            cls.pad_token,
            cls.unk_token,
            cls.sos_token,
            cls.eos_token,
        ] + list(unique_words)

        # 保存词表到文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            for word in vocab_list:
                f.write(word + "\n")

    @classmethod
    def from_vocab(cls, vocab_file):
        """
        从文件加载词表。

        :param vocab_file: 词表文件路径。
        :return: JiebaTokenizer 实例。
        """
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab_list = [line.strip() for line in f.readlines()]
        return cls(vocab_list)

    def __init__(self, vocab_list):
        """
        初始化 tokenizer。

        :param vocab_list: 词表列表。
        """
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        # 建立词到索引映射
        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        # 建立索引到词的映射
        self.index2word = {index: word for index, word in enumerate(vocab_list)}
        # 获取未知词索引
        self.unk_token_index = self.word2index[self.unk_token]
        self.pad_token_index = self.word2index[self.pad_token]
        self.sos_token_index = self.word2index[self.sos_token]
        self.eos_token_index = self.word2index[self.eos_token]

    def encode(self, sentence, add_sos_eos):
        """
        将句子编码为索引列表。

        :param sentence: 输入的句子。
        :return: 索引列表。
        """
        tokens = self.tokenize(sentence)
        # 在开始和结束位置添加开始和结束符号
        if add_sos_eos:
            tokens = [self.sos_token] + tokens + [self.eos_token]
        # 将 token 转为索引，未知词用 unk 索引替代
        return [self.word2index.get(token, self.unk_token_index) for token in tokens]


class ChineseTokenizer(BaseTokenizer):
    @classmethod
    def tokenize(cls, sentence) -> list[str]:
        # return jieba.lcut(sentence)
        return list(sentence)  # 既可以使用jieba分词，也可以使用直接把中文当分成单个字符


class EnglishTokenizer(BaseTokenizer):

    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()

    @classmethod
    def tokenize(cls, sentence) -> list[str]:
        return cls.tokenizer.tokenize(sentence)

    def decode(self, indexes):
        tokens = [self.index2word[index] for index in indexes]
        return self.detokenizer.detokenize(tokens)

    # @classmethod
    # def detokenize(cls, tokens: list[str]) -> str:
    #     return cls.detokenizer.detokenize(tokens)


if __name__ == "__main__":
    # 使用NLTK分词器对英文进行分词
    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()
    word_list = tokenizer.tokenize(
        "On a $50,000 mortgage of 30 years at 8 percent, the monthly payment would be $366.88."
    )
    print(word_list)
    print(detokenizer.detokenize(word_list))
    print(list("中文直接按单个字符分词"))
