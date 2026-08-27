# Task 13 TPU 实验 Runbook

**版本**：v1.1（v6e-16 多主机 TPU 支线；待训练前门控）  
**原始科学协议**：`/home/tanjunhao/Ego/Task13_Downstream_Policy_Matrix_Runbook.md` v1.1  
**TPU 操作参考**：`TPU_Spot_Operations_Guide.md`  
**状态**：计划与发射前门控；本文不授权训练。

## v1.1 优先修订：v6e-16 多主机支线

本节优先于本文旧版中关于“v6e-4、单主机、FSDP1、TPU 作为唯一正式
硬件”的表述。它反映当前已获分配的 Spot **v6e-16**（`us-east1-d`、
`4x4`、4 个 TPU VM worker）和本轮工作定位：TPU 是与正在运行的 A100
主线隔离的支线实验，不能替代、覆盖或计入 GPU 正式结论。

- 保留有科学含义的输入协议：Task13 两任务/五条件、冻结的 base、相机、
  norm、seed、global batch **32**、30,000 steps 和评测定义。不得修改
  canonical 数据或 GPU/A100 的 outputs。
- 允许为 v6e-16 使用 JAX 多进程/多主机 distributed mesh；不要求验证其与
  GPU 或单机 FSDP1 的逐数值等价，也不把该项作为本支线的发射条件。实际
  shard/FSDP 设置必须先在四 worker 的功能 smoke 中记录并冻结。
- 仍必须在发射前通过：四 worker 同时启动且看到全局 16 devices、真实资料
  可读、有限 loss，以及一次**多主机 checkpoint → GCS → 新本地副本恢复**。
  旧的单机 local-Orbax sync daemon 不能直接当作 v6e-16 的恢复方案；在
  多 worker save/restore 原型实际 PASS 前，不得发射 30k cell。
- 输入使用新的隔离前缀
  `gs://use1/user/tanjunhao/task13_tpu_sidebranch/v1/input_assets/`；输出及
  checkpoint 只能写入同一 `task13_tpu_sidebranch/v1/` 下按 run 隔离的前缀。
  旧 `task13_tpu_feasibility` 前缀不复用为本轮正式输入或输出。
- 30k 启动仍需用户明确确认；本轮预备工作、环境安装、输入复制和短功能
  smoke 均不构成正式训练启动。

## 0. 目的与不可变项

本 runbook 只把原始 Task13 的训练计算适配到 Spot Cloud TPU，不改变科学协议。

- 任务/条件不变：Assembly、Hammer；`nominal_src/repeat/visual/contact/combined`。
- 矩阵不变：每任务 13 个模型、共 26 个；training seeds `[42,43,44]`，nominal 仅 seed42。
- 训练配方不变：π0.5、Gemma 2B LoRA、Gemma 300M expert、global batch 32、30,000 steps、4 workers、原 optimizer/LR schedule/precision/action horizon/LoRA ranks、W&B disabled。
- 标准 config 仍是 `bimanual_assembly` 或 `hammer_nail`，不可使用 `*_rand_full`。
- Assembly 必须真实 `ego+wrist_left+wrist_right`、46-D state、44-D action、canonical 44-D base；Hammer 必须 `front+wrist`、23-D state、22-D action、原 π0.5 base。
- 每个 cell 必须完成 step29999，之后才可做原始 runbook 的 GPU/EGL 评测（156 runs / 7,800 episodes）。
- 不得因 TPU 改 batch、步数、相机、精度、数据、norm stats、seed、条件、评测或统计；不得因结果重训。

5090 仍是 Task13 数据、manifest、norm stats、最终报告及 GPU/MuJoCo/EGL 评测的 canonical owner。TPU 是纯训练计算资源，不承担正式评测。

## 1. 正式授权门槛

原 runbook 冻结的硬件是 5090、可选 A100；TPU 尚不是正式硬件。任何正式 step0 前必须同时满足：

1. 原始 Phase C、10 个 LeRobot 数据集、10 份 norm stats、原 runbook 要求的 smoke 和 prelaunch package 已 PASS。
2. 本文第 5 节 TPU qualification 全部 PASS。
3. 写出只读 superseding `training_hardware_allocation.json` 和 `training_launch_matrix.json`，将 TPU 定义为唯一正式训练硬件类别，并经用户/plan author 批准。
4. 收到明确 `GO_TRAIN_TPU`。

未收到 `GO_TRAIN_TPU` 时，只可做只读审计、环境/数据准备和已批准 smoke。26 个 cell 应使用同一 TPU 硬件类别；容量变化仅可调整未开始 cell 的 wave，不得将同 task×seed 的核心条件拆至不同硬件类别，亦不得跨硬件类别 resume。

## 2. 资源规格与调度

### 2.1 推荐规格

首选：**Spot `v6e-4`、`us-east1-d`、单主机 2×2 拓扑**。

- 先申请 **1 台** 用于 exact-formal qualification。
- qualification PASS 后，使用 **4–6 台独立 v6e-4 VM** 并发，每台每次仅一个 cell。
- 输入位于同区域 `gs://use1/user/tanjunhao/task13_tpu_feasibility/v1/input_assets/`，每个正式 run 用新的 output child prefix。
- 若控制台仅展示 64/320-chip 额度，应在额度内申请 4-chip VM；不要直接申请 64/320-chip multi-host Pod。

v5e-4 仅为 v6e-4 无容量时的后备，需全套重新 qualification。64-chip v4/v5e/v6e 和 320-chip v6e 通常为多主机 Pod，不适合当前冻结协议：multi-host loader/checkpoint 未验证，且不应以改变全局 batch32 来适配。

### 2.2 FSDP 规则

正式协议冻结 `fsdp_devices=1`。不得因为 v6e-4 有四片芯片就改为 FSDP=4；今天的 FSDP=4 技术 smoke 不是正式证据。

每个实际 TPU 类型必须先用 exact formal config 证明：single host、`jax.process_count()==1`、batch32、`fsdp_devices=1`、真实数据和相机均可通过。qualification 前不假设 FSDP=1 会高效使用所有芯片，也不在同一 VM 并行多个 JAX 训练进程；以实测吞吐、HBM、RAM、checkpoint 和磁盘数据决定是否发射。

## 3. Spot 生命周期、SSH 与身份

1. 创建时显式指定 `--project=whyu01`、zone、Spot 与批准的 accelerator type。
2. 每次新建/抢占后必须先 gcloud 首连：

   ```bash
   gcloud compute tpus tpu-vm ssh TPU_NAME --project=whyu01 --zone=ZONE
   ```

   它会下发 Compute Engine SSH key、known_hosts；不能先假定裸 SSH 可用。
3. 读取新 external IP，更新 `tanjunhao-tpu` SSH alias；Spot 重建后 IP 会变。
4. 记录 READY/health、type、topology、host/process 数、JAX devices、runtime、disk/RAM、attached service account。
5. 要求单主机且 `jax.process_count()==1`，否则停止。

优先用 TPU attached Compute Engine service account 的 bucket IAM。每次重建在专属 output prefix 做 GCS list/read 和小型 write/read/delete probe。只有 attached identity 不可用时，才从本机上传 scoped service-account key；密钥绝不进入 Git、GCS、日志、文档或聊天。

从 5090 操作 gcloud 时，网络异常可先运行 `proxy_on`；所有 gcloud 命令都带 `--project=whyu01`。

## 4. 代码、环境与输入

### 4.1 VM 外持久资产

- GitHub TPU 分支：`Tan20042023/Contact-Demo-Gen:task13-tpu-feasibility-prep`。
- GCS：版本化的只读 `input_assets`、按 run 隔离的 outputs。
- 5090：canonical Task13 数据、manifests、assets、base 与报告。

TPU 本地只放本 cell 输入 cache、独立环境、JAX cache、日志和临时 checkpoint。下载输入前后用 relative path、size、SHA-256 stable-tree manifest 验证，然后只读化；永远不向 `input_assets` 写入。

### 4.2 已验证环境基线

创建独立 Python 3.11 TPU environment，绝不复用 CUDA/GPU JAX environment：

```text
jax[tpu]==0.5.3
flax==0.10.2
orbax-checkpoint==0.11.13
torch==2.7.1
torchvision==0.22.1
torchcodec==0.5.*
lerobot==0.4.4
system ffmpeg
```

以 editable 安装 `openpi`，但不要安装仓库的 `jax[cuda12]` extra。验证 JAX devices/all-device collective、`import lerobot`、`import torchcodec`、`ffmpeg -version`。缺少系统 FFmpeg 会在第一个真实视频 batch 因 TorchCodec 失败；修复后重跑受影响 smoke。

若可选 `openpi-client` metadata 固定 NumPy1.26.4 而 TPU JAX/LeRobot 使用 NumPy2.x，不为消除该可选 client 的 `pip check` 告警而降级 TPU core stack；记录例外，验证实际训练导入路径。

## 5. TPU qualification gate

所有 smoke 仅写入独立 preflight/smoke output prefix，不能与正式 outputs 混用。

1. **Runtime gate**：JAX devices/collective、host-device transfer、attached identity、FFmpeg/TorchCodec、LeRobot decode、base local restore PASS。
2. **Checkpoint gate**：本地 Orbax save → GCS upload → fresh local download/restore PASS。
3. **Assembly exact smoke**：`bimanual_assembly/nominal_src/seed42`，真实三相机、44-D base、batch32、FSDP1、100 steps；finite metrics、checkpoint、policy-server load、44-D client inference PASS。
4. **Hammer exact smoke**：`hammer_nail/nominal_src/seed42`，原 base、batch32、FSDP1、100 steps；同样 finite/checkpoint/server/22-D client PASS。
5. 原 runbook 所需的每个实际 TPU task/condition smoke 覆盖必须齐全；单个 Hammer 技术 smoke 不能替代。
6. 记录 compile time、steady-state step time、HBM/chip、host RAM、input wait、checkpoint/GCS sync time、disk、code/package/base/data/norm SHA。

如 Assembly OOM、FSDP1/batch32/单主机不成立，或需要降低 batch/相机/精度、gradient accumulation、multi-host loader rewrite 或 JAX/Flax/Orbax core upgrade，停止并报告。

## 6. Spot-safe checkpoint 与恢复

Orbax0.11.13 不可靠地直接向 `gs://` 初始化空 temporary prefix：GCS object prefix 不是真目录。正式训练必须：

1. 在 TPU local disk 写 Orbax checkpoint；Orbax 原子发布数字 step 目录。
2. 并行运行 sync sidecar，只发现已出现的数字 step，`gcloud storage rsync` 到同一 GCS run prefix，比对 local/GCS 总字节数，再写 `UPLOAD_COMPLETE`。
3. resume 只接受含 `UPLOAD_COMPLETE` 的最大远端 step；无 marker 的对象一律视为不完整。
4. 退出时等待 Orbax async completion，再做一次最终 sync；记录 local/GCS bytes 和 marker path。

现有 helper：

```text
openpi/scripts/task13_tpu_checkpoint_sync_daemon.sh
openpi/scripts/task13_tpu_train_with_sync.sh
```

正式发射前必须审查并扩展它们以生成**原始正式 runbook 的 exact config**。不可误用早期 `task13_tpu_*` feasibility config（例如 FSDP4 或技术 checkpoint 策略）。

正式 checkpoint interval 先遵循原始协议的 10,000 steps；若希望改为 Spot 的 5,000，必须在 prelaunch review 明确批准 superseding protocol，并对全部26 cell一致地写入 launch manifest。

## 7. 正式发射与单 cell 完成

### 7.1 发射前 manifests

在 `GO_TRAIN_TPU` 后、任何正式 step0 前冻结：

- `training_hardware_allocation.json`：每 cell 的 TPU type/zone/VM、wave、hardware class=`cloud_tpu_v6e_4_spot`、关联 smoke SHA；全矩阵同类别。
- `training_launch_matrix.json`：task、condition、seed、data/assets/base SHA、exact config/command、local checkpoint root、GCS prefix、expected step29999。
- 每 VM 的 runtime、environment/pip list 与 code/package provenance。

建议按完整 task 或 task×seed block 排 wave。一个 VM 同时仅一个正式 cell。

### 7.2 完成条件

每 cell 必须：

- 从零开始，或只从同 cell、同 TPU hardware class、带 marker checkpoint resume；
- 完成30,000 steps，终点 step29999；
- terminal checkpoint local completion、GCS bytes match 与 `UPLOAD_COMPLETE` PASS；
- 最后 logged step≥29900，loss/grad/param norm finite，无 decoder/OOM/traceback；
- command、seed、data/norm/base/code SHA 与 launch manifest一致。

Spot preemption、crash、I/O、环境错误是基础设施失败，可修复后 exact-cell resume。step29999 可加载后，绝不因训练曲线或 eval result 重训。

## 8. GPU 评测、报告与清理

训练完成后，把带 marker 的 checkpoint 与对应 norm assets只读复制至原 runbook 的 GPU policy-server/EGL 节点，逐文件 size+SHA 验证。评测仍严格执行52个1-episode preflight、156正式 runs、7,800 episodes；不以 TPU 替代 GPU/MuJoCo/EGL。

TPU provenance 记录 VM/type/zone/Spot 状态，但不是新增科学因素。最终聚合、统计、Task14 freeze 与报告在5090 canonical root生成。

未获得用户 retention/cleanup 决定前，保留正式 checkpoint 与 manifests。获批准清理时，先 inventory 精确 local/GCS output prefix，再删除；绝不删除 input_assets、5090 canonical data、代码分支、bootstrap/environment docs 或本 runbook。

## 9. Hard stops

立即停止并报告，若出现：

- 未获 `GO_TRAIN_TPU`、未冻结 superseding hardware allocation；
- 多主机/多进程 slice，或 FSDP1 exact smoke/batch32失败；
- 需改变数据、norm、base、科学配方、GPU evaluation 或统计；
- Assembly 三相机/44-D 或 Hammer22-D schema失败；
- GCS identity、checkpoint marker/restore、stable-tree SHA 或 Spot recovery失败；
- 需跨 GPU/TPU hardware class resume，或结果可见后重新分配/重训；
- 需升级 JAX/Flax/Orbax core stack或改写multi-host loader才可运行。

## 10. 发射清单

- [ ] 原始Task13的 Phase C、conversion、norm stats、10个GPU smoke PASS。
- [ ] 正式 TPU allocation 批准；申请单主机 Spot v6e-4。
- [ ] input GCS版本/local cache/SHA/attached identity PASS。
- [ ] TPU environment（JAX/LeRobot/TorchCodec/FFmpeg）PASS。
- [ ] local Orbax → GCS marker → restore PASS。
- [ ] Assembly 与 Hammer exact-formal 100-step TPU smoke PASS。
- [ ] hardware/launch manifests冻结；收到 `GO_TRAIN_TPU`。
- [ ] 26/26 step29999 marked checkpoints complete。
- [ ] GPU 52 preflight + 156 eval runs + final Task13 analysis/freeze complete。
