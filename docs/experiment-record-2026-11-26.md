> 本文件托管在本仓库 docs/ 下，URL 稳定，可直接作为平台「训练记录链接」。
> 平台第二次试运行结束后，第 4 节会补入 hidden 指标与真实 Qwen 决策（文件名与链接不变）。
# 实验记录 · 智训引擎统一镜像与第二次平台试运行

日期：2026-11-26

| 项 | 值 |
| --- | --- |
| 镜像 | `zhixun-harness:ac-merged-cv2-final2-1126`（`sha256:f76234579195f0b6b303dad6ea7faac5473898646b75ef55070bc6b9f784d42b`，linux/amd64，SDK 1.0.4） |
| 源码 | https://github.com/gzy1234512/zhixun-harness-submission（commit `ab28c5f39918`） |
| 父模型 | 平台「训练起点」随包提供：A 4,326 / B 865 / C 833 参数 |
| 数据 | 平台公开投影：A 445 训练 / 1,244 验证；B 7,032 / 1,679；C 113,089 / 23,049 |
| 轮次 | A/B/C 各 10 轮 |
| 资源 | 2 CPU / 4 GiB / 无网络 / 只读根文件系统 |
| 付费调用 | 0（本地闭环用合成网关，只验证工程流程） |

本地闭环的模型回复由 `check_contest_104.py` 的合法固定策略给出（`CONTEST_104_VALID_FIXTURE=1`），
不调用任何付费 API；平台试运行的决策由真实 Qwen 给出，两者互不替代。

## 1. 读历史—做实验—依据结果决策

### A · 气体识别（Macro-F1 ↑）

| 轮 | 动作 | 父（同一投影实测） | 子（候选实测） | gain | 模型建议→下一步 | 实际选择 | 依据 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | decay_pair | 0.714030 | 0.880622 | +0.166592 | child→ratio_ensemble | accepted | 子优于父且达标 |
| 2 | ratio_ensemble | 0.880622 | 0.903189 | +0.022567 | child→adam_small | accepted | 子优于父且达标 |
| 3 | adam_small | 0.903189 | 0.908322 | +0.005133 | child→ratio_ensemble | accepted | 子优于父且达标 |
| 4 | ratio_ensemble | — | — | — | parent→adam_small | kept_parent | duplicate_of_parent |
| 5 | adam_small | 0.908322 | 0.910730 | +0.002407 | child→ratio_ensemble | accepted | 子优于父且达标 |
| 6 | ratio_ensemble | — | — | — | parent→adam_small | kept_parent | duplicate_of_parent |
| 7 | adam_small | 0.910730 | 0.911493 | +0.000764 | child→ratio_ensemble | accepted | 子优于父且达标 |
| 8 | ratio_ensemble | — | — | — | parent→adam_small | kept_parent | duplicate_of_parent |
| 9 | adam_small | 0.911493 | 0.910730 | -0.000764 | parent→adam_tiny | kept_parent | 未达接受条件 |
| 10 | adam_tiny | 0.911493 | 0.910730 | -0.000764 | parent→toluene_check | kept_parent | 未达接受条件 |

最终：accuracy 0.9694533762057878, acetaldehyde_n 109, acetaldehyde_tp 108, macro_f1 0.911493143861832

### B · 轴承寿命（RMSE ↓）

| 轮 | 动作 | 父（同一投影实测） | 子（候选实测） | gain | 模型建议→下一步 | 实际选择 | 依据 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | residual | 681.159833 | 658.982174 | +22.177659 | child→residual_light | accepted | 子优于父且达标 |
| 2 | residual_light | 658.982174 | 666.847782 | -7.865608 | parent→residual | kept_parent | 未达接受条件 |
| 3 | residual | 658.982174 | 658.982174 | +0.000000 | parent→adam_small | kept_parent | 未达接受条件 |
| 4 | adam_small | 658.982174 | 635.708127 | +23.274046 | child→residual | accepted | 子优于父且达标 |
| 5 | residual | 635.708127 | 658.294821 | -22.586694 | parent→residual_light | kept_parent | 未达接受条件 |
| 6 | residual_light | 635.708127 | 650.261629 | -14.553502 | parent→adam_small | kept_parent | 未达接受条件 |
| 7 | adam_small | 635.708127 | 640.410634 | -4.702507 | parent→adam_tiny | kept_parent | 未达接受条件 |
| 8 | adam_tiny | 635.708127 | 635.572113 | +0.136015 | child→residual | accepted | 子优于父且达标 |
| 9 | residual | 635.572113 | 646.872511 | -11.300399 | parent→residual_light | kept_parent | 未达接受条件 |
| 10 | residual_light | 635.572113 | 652.815786 | -17.243674 | parent→adam_small | kept_parent | 未达接受条件 |

最终：mae 519.9089966690503, rmse 635.5721125363481

### C · 发动机寿命（RMSE ↓）

| 轮 | 动作 | 父（同一投影实测） | 子（候选实测） | gain | 模型建议→下一步 | 实际选择 | 依据 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | c_bounded | 52.868573 | 47.661360 | +5.207212 | child→c_native_small | accepted | 子优于父且达标；风险门槛通过 |
| 2 | c_native_small | 47.661360 | 47.494269 | +0.167091 | child→c_bounded_narrow | accepted | 子优于父且达标；风险门槛通过 |
| 3 | c_bounded_narrow | 47.494269 | 48.897097 | -1.402828 | parent→c_bounded_shrink | kept_parent | 未达接受条件；风险门槛拒绝 |
| 4 | c_bounded_shrink | 47.494269 | 48.676375 | -1.182107 | parent→hold | kept_parent | 未达接受条件；风险门槛拒绝；下一步校验:planned_hold |
| 5 | hold | — | — | — | parent→hold | held | hold_by_policy；下一步校验:planned_hold |
| 6 | hold | — | — | — | parent→hold | held | hold_by_policy；下一步校验:planned_hold |
| 7 | hold | — | — | — | parent→hold | held | hold_by_policy；下一步校验:planned_hold |
| 8 | hold | — | — | — | parent→hold | held | hold_by_policy；下一步校验:planned_hold |
| 9 | hold | — | — | — | parent→hold | held | hold_by_policy；下一步校验:planned_hold |
| 10 | hold | — | — | — | parent→hold | held | hold_by_policy；下一步校验:planned_hold |

最终：mae 33.75297974682287, nasa_score 1548436542.8351073, rmse 47.4942687751518

## 2. C 全量预测计时：定位与修复

上一版镜像在本地主机上 C 全量预测 **108.34 秒**，超过主机 100 秒限制，集成报告无法生成。
容器内 cProfile（600 行）显示每行 3 次 `lstat` + 6 次 `stat`，占 2.632 秒中的 86%；树推理本身只有 34 毫秒。
`/model` 是绑定挂载（本机 9p/drvfs），元数据调用比本地磁盘慢两个数量级。

修复：`contest_submission.predict_target` 把挂载产物校验改为每进程一次（键为挂载目录与 `CONTEST_ARTIFACT_PATH`）。
校验语义不变：软链接、越界路径、扩展文件存在性、描述符不符仍然拒绝，并新增 6 条契约测试。

| 口径 | 修复前 | 修复后 |
| --- | --- | --- |
| 容器内文件管道全量预测 | 108.34 s | 4.11 s |
| 本地 host 测量（100 秒限制） | 105.63 s（超时） | 4.02 s |
| 23,049 行输出 | `38899d516ab14e8d` | 逐字节相同 |

未放宽任何超时：`check_contest_104.py` 仍是 40 秒首行 / 100 秒全量。

## 3. 统一镜像验收

| 项目 | 结果 |
| --- | --- |
| A 10 轮 fixture | Macro-F1 0.911493143862 |
| B 10 轮 fixture | RMSE 635.572113 |
| C 10 轮（4 个动作后用尽，后续 hold） | RMSE 47.4942688 |
| C 4 轮合成网关集成 | passed=True，动作 c_bounded, c_native_small, c_native_small, c_bounded_narrow |
| C 全量公开验证 | 23049 行 3.9974 秒，RMSE 47.4942688 |
| A/B 合并前后回归 | 预测与一次原生续训逐项相同 |
| 单测 | A 19、C 11、live 5、SDK 18 |

## 4. 平台第二次试运行（2026-11-26，待填）

> 平台报告下载后补：每次模型调用的建议与实际选择、每轮 hidden 指标、A/B/C 最终 hidden 成绩、
> 与第一次试运行（A 0.760658 / B 685.318873 / C 55.237329 / NASA 8.030837437e10）的对照。
