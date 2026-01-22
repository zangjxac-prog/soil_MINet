# soil_MINet

该项目提供一个将微生物属丰度序列视为“句子”的 GPT-2 风格语言模型训练示例。主要思路：

- 每个样本是一条序列，序列中的 token 为“微生物属”。
- 按照丰度从高到低排序生成 token 顺序，对应模型的位置编码。
- 使用 GPT-2 类似的 Transformer 解码器结构进行自回归语言模型训练。

## 数据格式

支持两种输入格式：

### JSONL

每行一个样本，内容为 `{属: 丰度}` 的字典：

```json
{"Bacillus": 0.31, "Pseudomonas": 0.12}
```

### CSV（宽表）

表头为属名，每一行代表一个样本，可选第一列 `sample_id`：

```csv
sample_id,Bacillus,Pseudomonas
S1,0.31,0.12
S2,0.05,0.43
```

## 训练

```bash
python -m soil_minet.train \
  --data-path data/abundance.jsonl \
  --output-dir outputs \
  --block-size 256 \
  --batch-size 16 \
  --epochs 5
```

训练完成后会输出：

- `outputs/gpt_model.pt`：模型权重
- `outputs/vocab.json`：词表（属 -> token id）
- `outputs/config.json`：模型配置

## 训练逻辑说明

- 使用 `soil_minet.data.encode_sample` 将样本按丰度排序并生成 token 序列。
- 使用 `soil_minet.model.GPTModel` 进行 GPT-2 风格的自回归训练。

如需调整模型层数、头数或嵌入维度，可通过命令行参数 `--n-layers`、`--n-heads`、`--n-embd` 控制。
