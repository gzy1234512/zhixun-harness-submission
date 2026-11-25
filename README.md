# 智训引擎实验复现

本目录对应使用 Harness SDK 1.0.4 的三任务闭环实验。首轮加载 A、B、C 各自的官方初始检查点，每轮训练并评价候选，根据决策接受候选或保留父模型。A 使用 Macro-F1，B 使用分钟 RMSE，C 使用周期 RMSE。实验以公开验证数据评分。

## A 项目的甲苯复核

A 项目先试 `decay_pair`，用乙醇和乙醛样本训练两类纠错。采用该候选后，可以试 `toluene_check`。它保留父模型的纠错参数，使用本轮授权训练集的甲苯和丙酮样本训练单独的二分类器。推理时只复核预测为甲苯的样本，二分类器判为丙酮时才改判。

平台用同一阶段的公开验证数据对比父模型与候选。候选 Macro-F1 提高后才有资格采用，选中后随 `extension.npz` 加载到后续轮次和预测流程。

## 环境

使用 Ubuntu 24.04 x86-64 或 Windows 上的 WSL2 Ubuntu 24.04。准备可从 Linux 终端访问的 Docker，以及 Python 3.12 和 `python3.12-venv`。以下命令在本目录根路径执行。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --no-index --find-links docker/wheels \
  --find-links docker/v17-training-wheels -r requirements-contest.lock
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
docker build --platform linux/amd64 -f Dockerfile.contest-base -t zhixun-python:ubuntu24 .
docker build --network none --platform linux/amd64 \
  --build-arg PYTHON_IMAGE=zhixun-python:ubuntu24 -f Dockerfile.contest \
  -t zhixun-harness:mentor-abc-1125 .
```

首次构建基础镜像需要访问 Ubuntu 软件源。训练容器使用目录中附带的依赖包。

## 运行

离线验证使用固定模型回复，执行真实工业模型训练、公开验证和独立预测，无需 API Key。

```bash
python -m scripts.check_contest_104 --mode fixture \
  --run-id contest-104-fixture-abc --tasks ABC --rounds 1 \
  --image zhixun-harness:mentor-abc-1125
```

真实实验使用 `Qwen/Qwen3.8-27B`，调用会产生费用。用户已于 2026-11-25 确认本项目累计消耗为 0，本版已加入 `runs/v12-trainer/cost-ledger.json`，初始累计金额为 0，调用记录为空。账本的 30 元累计限额沿用原代码，费用按代码内置单价估算，实际费用以 PyTRIO 账单为准。运行后程序会持续更新这份账本，后续实验应保留已有记录。

先在 Ubuntu 终端进入 1125 版项目并激活环境。

```bash
cd -P '/mnt/e/ai比赛/zhixun1125/experiment-runtime'
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

单独运行下面一行，看到提示后粘贴 PyTRIO API Key 并回车。输入时终端隐藏密钥内容。

```bash
read -rsp '请粘贴 PyTRIO API Key：' PYTRIO_API_KEY && printf '\n' && export PYTRIO_API_KEY
```

在同一个终端运行 A 任务的一轮真实 Qwen 实验。

```bash
CONTEST_104_PYTRIO_TRANSPORT=openai python -m scripts.check_contest_104 --mode live \
  --run-id contest-104-live-a-1125 --tasks A --rounds 1 \
  --image zhixun-harness:mentor-abc-1125
```

成功结束时终端会显示 `"passed": true` 和 `"paid_request_count": 1`。汇总报告保存到 `reports/contest-submission-1.0.4/contest-104-live-a-1125.json`。其中 `proposal` 记录 Qwen 的建议，`selection` 记录实际采用结果。

三任务各运行 10 轮时使用下面的独立运行编号。

```bash
python -m scripts.check_contest_104 --mode live \
  --run-id contest-104-live-abc --tasks ABC --rounds 10 \
  --image zhixun-harness:mentor-abc-1125
```

也可使用 PyTRiO SDK 采样。先运行 `python -m pip install -r requirements-v12-trainer.lock`，再设置 `CONTEST_104_PYTRIO_TRANSPORT=sdk`。Tokenizer 已放在宿主脚本的默认路径 `runs/v12-preflight/tokenizer/`。

运行记录和各轮选中的检查点写入 `runs/<run-id>/`，汇总指标写入 `reports/contest-submission-1.0.4/<run-id>.json`。中断后用同一镜像和命令恢复。另开实验时使用新的 `run-id`。

## 源码与数据

- `contest_submission.py`、`contest_engine/` 实现训练、选择和预测。
- `scripts/check_contest_104.py` 驱动本地实验，`scripts/contest_live_gateway.py` 接入真实 Qwen。
- `vendor/reference/` 保存三任务的公开训练与验证数据及初始 `checkpoint.npz`。

SDK 回归测试可在本目录这样运行。

```bash
(cd vendor/contest-harness-sdk-1.0.4 && \
  python -m unittest tests.test_sdk tests.test_contract_regressions tests.test_transport_regressions -v)
```
