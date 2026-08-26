# Hand by Hand

本仓库使用 [uv](https://docs.astral.sh/uv/) 统一管理根目录下的
`transformers` 和 `detr` 两个学习项目。

## 安装依赖

```bash
uv sync
```

`uv sync` 会根据 `pyproject.toml` 和 `uv.lock` 创建或更新根目录下的
`.venv`。运行模块时使用：

```bash
uv run python -m detr.check_data
uv run python -m detr.train
uv run python -m transformers.train
```

## 增删依赖

不要再使用 `pip install` 修改项目环境。新增运行依赖时使用：

```bash
uv add <package>
```

新增开发依赖时使用：

```bash
uv add --dev <package>
```

删除依赖时使用：

```bash
uv remove <package>
```

`pyproject.toml` 记录直接依赖，`uv.lock` 锁定完整依赖树；这两个文件都应提交到 Git。
