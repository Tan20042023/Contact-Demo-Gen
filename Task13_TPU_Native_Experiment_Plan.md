# Task13 TPU 原生实验方案

**版本**：v1.1（2026-08-27）  
**定位**：Spot TPU 支线；与 A100/GPU 正式主线隔离，不替代其结论。  
**目标**：在不改动 canonical 输入、base、相机 schema 与 GPU 输出的前提下，
让 Spot `v6e-16` 从拿到资源到可恢复训练的时间尽可能短。

## 1. 决策

- 立即资源：一块 Spot `v6e-16`，`us-east1-d`，4 个 worker、16 chips。
- 扩容策略：第一块通过恢复验证后，如需吞吐，增加第二块 **独立的**
  `v6e-16`，各跑一个独立 cell；不把单个 cell 扩到 `v6e-32`。
- 每个 cell 仍使用 global batch 32（2 samples/chip）。这是有意保留的
  优化稳定性锚点，而非沿用 GPU 的拓扑限制。`v6e-32` 会把它降至
  1 sample/chip，同时增加到 8 host，当前模型和 input pipeline 不值得承担
  这份通信、恢复与调试成本。
- 允许并冻结 v6e-16 的多主机 JAX mesh/sharding 实现；不要求与 GPU、
  单主机或 FSDP1 逐数值对齐。所有 TPU 结果标为 `tpu_sidebranch`。

## 2. 实验矩阵：分层、预先定义

这是探索性 TPU 方案，不以“把 GPU 26-cell 规范原样搬来”为目标。

| 阶段 | 预先定义的工作 | 目的 | 是否是长训练 |
| --- | --- | --- | --- |
| P0 | 每 task 一个 100-step 真数据 smoke + 一次全量恢复 | 发现 topology、decoder、checkpoint 问题 | 否 |
| P1 | 两 task 的 `nominal_src`、seed 42，各 1,000 steps | 量化 compile、吞吐、2,500-step checkpoint 开销 | 否 |
| P2 | 2 tasks × 5 conditions × seed 42 = 10 个 30k cells | TPU 支线的主筛查矩阵 | 是 |
| P3（可选，预定义） | P2 的全部非-nominal cells 用 seed 43、44：16 个 30k cells | 增加统计把握，不按 P2 结果挑选 cell | 是 |

P2 已覆盖所有条件且结果可解释；P3 不是 P2 的启动前置条件。若执行 P3，
必须完整执行预定义的 16 个补充 cell，不能依据曲线或评测结果只挑“好看”的
cell。GPU 主线仍保留其自己的正式矩阵与评测规则。

## 3. Spot 快速发射设计

所有持久资产放在同区域的 GCS，结构固定为：

```text
gs://use1/user/tanjunhao/task13_tpu_sidebranch/v1/
  input_assets/       # immutable datasets, bases, norms, manifests
  bootstrap/<release>/ # startup script, source archive, pinned wheelhouse, manifest
  runs/<campaign>/<cell>/
    checkpoints/<step>/worker-<index>/
    COMMITTED.json
    logs/
    provenance/
```

每次 TPU 创建时使用 `startup-script`。该脚本只能做下列无训练副作用的事：

1. 安装系统依赖（Python 3.11、FFmpeg）。
2. 拉取已版本化的 code archive 和 Python wheelhouse；离线创建 TPU venv。
3. 并行下载/校验输入 cache，挂载为只读。
4. 写出每个 worker 的 `READY.json`（runtime、release SHA、磁盘、GCS identity）。

**启动脚本绝不启动训练。** 任意已认证的控制端（本机或 Cloud Shell 优先）
只在四份 `READY.json`、GCS probe 与 multi-host collective 都 PASS 后，才
接受明确的发射命令。5090 不属于运行时依赖，也不保存训练必需资产。

一次性建设后，常规重建路径应是“创建 TPU → GCS bootstrap 四机并行就绪 →
preflight → launch”，而不再临时安装、临时 clone、临时找输入。启动时长以实际
时间记录为准；本方案不承诺未经测量的分钟数。

### 3.1 可选的持久输入盘

若后续频繁抢占，建立一个 `us-east1-d` 的 **Hyperdisk ML / PD** 只读输入盘，
存放已校验的 `input_assets` 和 bootstrap release。多 host TPU 可以把这类盘
只读挂到每个 worker；它减少四份大文件重复下载，但不承担 checkpoint 或可写
输出。该资源有独立费用，只有在用户批准创建持久盘后实施。

## 4. 多主机训练与恢复契约

单机时期的 Orbax local-sync daemon 不能直接复用。v6e-16 的实现必须先通过：

1. 四个进程同时 `jax.distributed.initialize()`，确认 `process_count=4`、
   `device_count=16`，并完成跨 host collective。
2. 检查实际 Orbax 输出，明确每个 process 写出的文件；不得臆测文件布局。
3. 每个 process 将自己的已完成 shard 上传到
   `checkpoints/<step>/worker-<index>/`，并写 sizes/SHA manifest。
4. 仅 worker 0 在收到四份正确 manifest 后写 `COMMITTED.json`。它记录 step、
   process count、release/config/base/data SHA、每个 worker 的字节数与 SHA。
5. 恢复时只认可含 `COMMITTED.json` 的最大 step；四个 worker 分别下载自己的
   shard，屏障同步后实际 restore，继续至少一个训练 step。

2026-08-27 的 v6e-16 结果：四机 collective、分片 data loader 与 Hammer
`nominal_src` 的 100-step 真训练通过（稳态约 1.3 step/s）；在隔离的 1-step
checkpoint 保存期间 Spot 被抢占，未产生 `COMMITTED.json`。因此 checkpoint
契约仍是 P0 的阻塞门，不能以“训练已跑通”替代。

checkpoint interval 先设为 **2,500 steps**，并在 P1 记录 save+upload 时间。
理由是 Spot 下平均最多损失约 1,250 steps，较原 5,000 更适合初期不稳定的
容量；若 P1 证实保存成本过高，可在 P2 前统一改为 5,000，不能在同一阶段中
随意改变。

## 5. 发射与扩容规则

- 一个 `v6e-16` slice 同时只运行一个 cell；每个 cell 有独立 run prefix、
  日志和 checkpoint manifest。
- 只有 P0/P1 通过后才启动 P2。P2/P3 的 start/stop 由用户明确批准。
- 一块 slice 被抢占：不等它恢复；新建同规格 slice，自动 bootstrap，然后从
  最近的 `COMMITTED.json` 恢复。
- 需要吞吐时，第二块 `v6e-16` 跑另一个 cell。不得把同一个 cell 的 checkpoint
  移到不同 topology 或从 GPU checkpoint 恢复。
- 不使用 `v6e-32`，除非以后专门完成 batch/mesh scaling 实验并显示它对该模型
  的实际稳态吞吐显著优于两块独立 v6e-16。

## 6. 不变的科学和安全边界

- 不修改 5090 的 canonical data、norm、base、A100/GPU outputs 或环境。
- 不把 TPU 结果与 GPU 正式结果混合统计；评测需要时仍在 GPU/MuJoCo/EGL 执行。
- P2/P3 的 data/base/norm/camera/action schema 冻结，并随每个 cell 保存 provenance。
- Spot VM、local disk 和未提交 checkpoint 一律视为可丢失；Git、GCS input、
  `COMMITTED.json` checkpoint 是唯一恢复来源。

## 7. 近期执行顺序

1. 完成并封存 GCS `input_assets` 上传与 checksum 验证。
2. 在任意已认证控制端构建并发布一个 GCS bootstrap release（script、source archive、wheelhouse、input manifest）。
3. 创建下一块 v6e-16 时传入 startup script；等待四机 READY。
4. 实现/验证多主机 checkpoint contract，运行 P0/P1。
5. 向用户报告实测启动、compile、吞吐、save/upload/restore 时间，再请求 P2 发射确认。
