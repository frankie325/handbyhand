# 使用代理模式（直接传递类）重构数据集构建逻辑

本计划旨在通过将数据集类作为参数直接传递，重构 `detr/datasets/build.py` 中的构建逻辑，以消除硬编码的条件分支，提高代码的可扩展性。

## 当前状态分析
- `build_dataset` 和 `build_dataloader` 函数目前使用 `dataset_type` 字符串并通过 `if-elif` 分支来决定实例化哪个数据集类（`VocDetection` 或 `RaodDetection`）。
- 这种方式在增加新数据集时需要修改核心构建逻辑，不符合开闭原则。

## 拟议变更

### 1. 修改 `detr/datasets/build.py`
- 重构 `build_dataset` 函数：
    - 将 `dataset_type: str` 参数替换为 `dataset_cls: type[Dataset]`。
    - 移除 `if-elif` 分支，直接调用 `dataset_cls(...)` 进行实例化。
- 重构 `build_dataloader` 函数：
    - 同步修改参数，将 `dataset_type: str` 替换为 `dataset_cls: type[Dataset]`。
    - 在调用 `build_dataset` 时传递 `dataset_cls`。

### 2. 增强 `detr/datasets/__init__.py`
- 为了保持配置的灵活性，在此处维护一个 `DATASET_REGISTRY` 字典，将字符串映射到对应的类。
- 这样可以在 `train.py` 等入口处根据配置字符串快速获取对应的“类对象”。

### 3. 更新调用端代码
- **`detr/train.py`**:
    - 根据 `config.DATASET_TYPE` 从 `DATASET_REGISTRY` 获取对应的类。
    - 将获取到的类对象传递给 `build_dataloader`。
- **`detr/validate.py`**:
    - 执行相同的重构逻辑，确保验证过程也能正确加载数据集。

## 假设与决策
- **决策**: 采用用户确认的“直接传递类”方案。这意味着构建函数不再负责“根据名字找类”，而是由调用者（或配置代理层）决定使用哪个类。
- **假设**: 所有数据集类（`VocDetection`, `RaodDetection` 等）的构造函数接口保持一致（`root`, `image_set`, `transforms`）。

## 验证步骤
- **代码检查**: 确保 `build_dataset` 中不再包含 `if dataset_type == ...` 逻辑。
- **功能测试**: 运行 `python -m detr.train` 的初始化部分，确认能够正确根据 `config.py` 中的 `raod` 配置加载 `RaodDetection` 实例。
- **类型检查**: 确保传递的是类对象本身，而不是字符串或实例。
