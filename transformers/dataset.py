from .config import PROCESSED_DATA_DIR, BATCH_SIZE
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence


class TranslationDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_json(file_path, lines=True).to_dict("records")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        input_tensor = torch.tensor(item["zh"], dtype=torch.long)
        target_tensor = torch.tensor(item["en"], dtype=torch.long)
        return input_tensor, target_tensor


# !将一批中的序列长度不一致的序列填充为相同长度
def collate_fn(batch):
    # batch: 二元组列表 [(input_tensor, target_tensor), (input_tensor, target_tensor), ...]
    inputs, targets = zip(*batch)
    # print("inputs:", inputs)
    # pad_sequence方法从一批中找到最长的序列长度，然后用0填充剩余的序列，再堆叠成[batch_size, seq_len]的长度
    inputs = pad_sequence(list(inputs), batch_first=True, padding_value=0)
    # print("stack inputs:", inputs)
    targets = pad_sequence(list(targets), batch_first=True, padding_value=0)
    return inputs, targets


def get_dataloader(train=True):
    file_path = (
        PROCESSED_DATA_DIR / "train.jsonl"
        if train
        else PROCESSED_DATA_DIR / "test.jsonl"
    )
    dataset = TranslationDataset(file_path)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    return dataloader


if __name__ == "__main__":
    # 测试数据集和数据加载器
    train_loader = get_dataloader(train=True)
    test_loader = get_dataloader(train=False)
    print(f"Number of batches in training set: {len(train_loader)}")
    print(f"Number of batches in testing set: {len(test_loader)}")
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"Inputs: {inputs}")
        print(f"Targets: {targets}")
        print("shape of input:", inputs.shape)  # [batch_size, seq_len]
        print("shape of target:", targets.shape)
        break  # 只打印第一个批次
