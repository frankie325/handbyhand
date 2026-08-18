import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
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
    WARMUP_STEPS,
    CLIP_GRAD_NORM,
    MODELS_DIR,
)
import time
from .dataset import get_dataloader
from .model.make import make_model
from .tokenizer import ChineseTokenizer, EnglishTokenizer
from tqdm import tqdm
from .utils import make_std_mask


def train_one_epoch(
    model, data_loader, loss_fn, optimizer, scheduler, device, zh_tokenizer, en_tokenizer
):
    model.train()
    total_loss: float = 0.0
    for inputs, targets in tqdm(data_loader, desc="Training"):
        # inputs: [batch_size, src_len] # src_len中文序列长度
        # targets: [batch_size, tgt_len] # tgt_len英文序列长度
        inputs = inputs.to(device)
        targets = targets.to(device)

        # src_mask: 屏蔽源序列中的 padding 位置
        # [batch_size, src_len] -> [batch_size, 1, src_len]
        src_mask = (inputs != zh_tokenizer.pad_token_index).unsqueeze(-2)

        # 目标序列做 teacher forcing 的左右移位：
        # - decoder 输入 tgt_in  = targets[:, :-1]：去掉最后一个 token，（<eos> 是解码器最后一步要预测的，不当作输入）
        tgt_in = targets[:, :-1]
        # - 预测目标 tgt_out    = targets[:, 1:]：去掉开头的 <sos>
        tgt_out = targets[:, 1:]

        # tgt_mask 必须基于 tgt_in 生成
        tgt_mask = make_std_mask(tgt_in, en_tokenizer.pad_token_index)

        # 前向传播，decode_outputs: [batch_size, tgt_len-1, d_model]
        decode_outputs = model(inputs, tgt_in, src_mask, tgt_mask)

        # !经过检测头进行预测
        outputs = model.generator(
            decode_outputs
        )  # [batch_size, tgt_len-1, en_vocab_size]

        # 损失计算
        loss = loss_fn(
            outputs.contiguous().view(-1, outputs.size(-1)),
            tgt_out.contiguous().view(-1),
        )

        # 反向传播 + 梯度裁剪 + 参数更新
        optimizer.zero_grad()
        loss.backward()
        # 防止梯度爆炸导致训练不稳定
        if CLIP_GRAD_NORM is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD_NORM)
        optimizer.step()
        # Noam 学习率调度：每个 batch 更新一次学习率
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)  # 返回平均损失


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("设备:", device)

    # tensorboard 记录，终端切换到当前目录下，输入tensorboard --logdir=logs
    writer = SummaryWriter(log_dir=LOG_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))

    # 加载数据
    data_loader = get_dataloader()

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

    """
    !忽略填充的token的损失计算: pad不需要参与损失计算 
    Decoder 输入：[s] [] [] []  []  [e] 
                                |   | 
    Decoder 输出：[s] [] [] [e] [p] [p] 
    """
    # 注意：Generator 输出的是 log_softmax（见 transformer.py），
    # 所以这里必须用 NLLLoss，而不是 CrossEntropyLoss——
    # CrossEntropyLoss 内部自带 log_softmax，若再配合 log_softmax 输入
    # 会"双重 softmax"，导致损失计算错误、模型不收敛。
    loss_fn = nn.NLLLoss(ignore_index=en_tokenizer.pad_token_index)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Noam 学习率调度（Transformer 论文标准做法）：
    # lr = D_MODEL^(-0.5) * min(step^(-0.5), step * WARMUP_STEPS^(-1.5))
    # 前 WARMUP_STEPS 步线性爬升，之后按 1/sqrt(step) 衰减
    def lr_lambda(step: int) -> float:
        step = max(step, 1)
        return (D_MODEL**-0.5) * min(
            step**-0.5, step * (WARMUP_STEPS**-1.5)
        ) / LEARNING_RATE

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(
            model,
            data_loader,
            loss_fn,
            optimizer,
            scheduler,
            device,
            zh_tokenizer,
            en_tokenizer,
        )
        # tensorboard 记录
        writer.add_scalar("Loss/train", avg_loss, epoch)
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch, EPOCHS, avg_loss))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODELS_DIR / "best.pth")
            print("模型已保存到:", MODELS_DIR / "best.pth")

    writer.close()


if __name__ == "__main__":
    train()
