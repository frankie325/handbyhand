import torch
from .config import (
    LOG_DIR,
    PROCESSED_DATA_DIR,
    N_LAYER,
    D_MODEL,
    D_FF,
    N_HEAD,
    DROPOUT,
    MAX_SEQ_LEN,
    LEARNING_RATE,
    EPOCHS,
    MODELS_DIR,
)
from .model.make import make_model
from .tokenizer import ChineseTokenizer, EnglishTokenizer
from .utils import make_std_mask


def predict_batch(inputs, model, zh_tokenizer, en_tokenizer):

    # input.shape: [batch_size, src_seq_len]
    # return [(*,*,*,eos), (*,*,*,*,..., eos),...]

    # 模型预测
    model.eval()
    with torch.no_grad():
        # 编码
        # src_mask: 屏蔽源序列中的 padding 位置
        # [batch_size, src_len] -> [batch_size, 1, src_len]
        src_mask = (inputs != zh_tokenizer.pad_token_index).unsqueeze(-2)

        memory = model.encode(
            inputs, src_mask
        )  # memory.shape: [batch_size, src_seq_len, dim_model]

        # 解码
        batch_size = inputs.shape[0]
        decoder_input = torch.full(
            [batch_size, 1],
            en_tokenizer.sos_token_index,
            dtype=torch.long,
            device=inputs.device,
        )  # [batch_size, tgt_len] 第一个时间步时的输入是<sos>，tgt_len=1

        """
        记录每个样本是否已经生成结束符

        预测结果：
        [] [] [] []    [eos] []
        [] [] [] [eos] []    []
        [] [] [] []    []    [eos]

        is_finished存储的是每个样本是否已经生成结束符
        [第一个样本是否预测结束, 第二个样本是否预测结束, ...]
        """
        is_finished = torch.full([batch_size], False, device=inputs.device)

        # 自回归生成
        generated = []  # 预测结果缓存
        for i in range(MAX_SEQ_LEN):
            tgt_mask = make_std_mask(decoder_input, en_tokenizer.pad_token_index)

            decoder_output = model.decode(
                memory, src_mask, decoder_input, tgt_mask
            )  # decoder_output.shape: [batch_size, tgt_len, en_vocab_size]

            # !保存预测结果，从decoder_output取出最后一个序列就是该时间步的预测词
            next_token_indexes = torch.argmax(
                decoder_output[:, -1, :],  # [batch_size, en_vocab_size]
                dim=-1,
                keepdim=True,  # dim=-1会降维，设置keepdim保持形状为[batch_size, 1]
            )  # !从vocab_size个词里面拿到概率最大的索引就是预测词的索引  shape: [batch_size, 1]
            generated.append(next_token_indexes)

            # !更新输入：将预测词和之前的预测词拼接起来，作为下一个时间步的输入
            decoder_input = torch.cat([decoder_input, next_token_indexes], dim=-1)

            # 判断是否应该结束
            """
            示例：
            假设预测结果：3个样本
            [] [] [] []    [eos] []
            [] [] [] [eos] []    []
            [] [] [] []    []    [eos]

            第一个时间步：is_finished = [false, false, false]
            第二个时间步：is_finished = [false, false, false]
            第三个时间步：is_finished = [false, false, false]
            第四个时间步：is_finished = [false, true, false]
            第五个时间步：is_finished = [true, true, false]
            第6个时间步：is_finished = [true, true, true]
            """
            is_finished = is_finished | (
                next_token_indexes.squeeze(1) == en_tokenizer.eos_token_index
            )

            # 全部预测结束，跳出循环
            if is_finished.all():
                break

        """
        处理预测结果 
        [第一个时间步的预测结果, 第二个时间步的预测结果, ...]
        [tensor(batch_size, 1), tensor(batch_size, 1), ...]
        """
        generated_tensor = torch.cat(generated, dim=1)  # [batch_size, 预测序列长度]
        generated_list = generated_tensor.tolist()  # [[*,*,*,eos,*], [*,*,*,*,eos],...]

        # 去掉eos之后的token_index
        for index, sentence in enumerate(generated_list):
            if en_tokenizer.eos_token_index in sentence:
                eos_pos = sentence.index(en_tokenizer.eos_token_index)
                generated_list[index] = sentence[:eos_pos]

        return generated_list


def predict(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("设备:", device)

    zh_tokenizer = ChineseTokenizer.from_vocab(PROCESSED_DATA_DIR / "zh_vocab.txt")
    en_tokenizer = EnglishTokenizer.from_vocab(PROCESSED_DATA_DIR / "en_vocab.txt")

    model = make_model(
        zh_tokenizer.vocab_size,
        en_tokenizer.vocab_size,
        N_LAYER,
        D_MODEL,
        D_FF,
        N_HEAD,
        DROPOUT,
        MAX_SEQ_LEN,
    ).to(device)

    model.load_state_dict(torch.load(MODELS_DIR / "best.pth", map_location=device))

    indices = zh_tokenizer.encode(text, False)
    input_tensor = (
        torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    )  # [1, tokens长度]

    # 模型预测
    batch_result = predict_batch(input_tensor, model, zh_tokenizer, en_tokenizer)

    res = en_tokenizer.decode(batch_result[0])

    return res


if __name__ == "__main__":
    # 测试预测函数
    test_text = "我喜欢你"
    result = predict(test_text)
    print(f"输入: {test_text}")
    print(f"预测结果: {result}")
