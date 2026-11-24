# Inkling-2-Mini TP4 并行运行适配

本目录以 stock SGLang + DSpark 在四台 DGX Spark 上运行
`thinkingmachines/Inkling-2-Mini`，可与现有 `deepseek` TP4 target 并行。

## 固定边界

- API：`0.0.0.0:8889`，容器名 `inkling-mini-r0..3`
- SGLang dist-init：`spark-b` 管理地址 `192.168.2.161:25001`
- NCCL control：`:25002`
- 模型：`/opt/models/thinkingmachines/Inkling-2-Mini`
- 缓存：`/opt/runtime_assets/inkling-mini/cache`
- 镜像来源：`lmsysorg/sglang@sha256:bc811ac7...`，仅 `linux/arm64`。
  逐跳 `docker save | docker load` 不保留 registry digest 引用，因此四机
  以官方构建 tag `lmsysorg/sglang:nightly-dev-20261120-03fde9cc` 消费，
  并在启动前逐机核对 image ID
  `sha256:2e4eb06d...01c7ce`；digest 仍作为来源 provenance 登记。
- 环网：父仓 `ring/nccl-ringonly-v5-2.30.7`，`NCCL_RING_DEV_MAP`
  由 `config.yaml.common.ring` 派生

## TP4 口径

- `nnodes=4`、`tensor_parallel_size=4`、`ep_size=4`；容器实际参数为
  `--tp 4 --ep 4`，`/get_server_info` 也返回 `tp_size=4`、`ep_size=4`。
- 4 x 256K：`context_length=262144`、`max_request=4`，总 KV 逻辑容量
  约 1M。
- `mem_fraction_static=0.73`、`swa-full-tokens-ratio=0.04`、
  `max_prefill_tokens=4096`、`chunked_prefill_size=4096`，容器内存上限
  `20g`。
- 当前实测 KV pool：`1,073,024` tokens，覆盖 4 个 256K 请求并保留
  24,448 tokens 余量；`available_gpu_mem=5.03 GB`。启动后主机
  `MemAvailable` 约 3.65-5.26 GiB。
- `max_mamba_cache_size=20`：Inkling2 每个请求需要 5 个 mamba state
  slot；4 并发至少需要 20。
- 当前挂载 `overlay/` 的 prefill/attention/pfuse 性能栈；DSpark、
  decode CUDA graph 和 prefill CUDA graph 均开启。

## 当前性能基线

2026-11-24，DeepSeek 6 路并行运行，Inkling TP4，prose decode 使用
speedbench 0.3 的 18 条 prose prompt，256 output tokens，T=0：

| prose | c1 | c2 aggregate | c4 aggregate |
|---|---:|---:|---:|
| tok/s | 67.1 | 101.6 | 149.7 |

cold prefill，speedbench longctx，3 reps：

| context | 8K | 16K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|
| prefill tok/s | 4,615 | 5,082 | 5,424 | 5,217 | 4,745 |

## 并行启动

```bash
cd /opt/spark
./deploy.sh start --target=inkling-mini \
  --model=thinkingmachines/Inkling-2-Mini
```

`start --target=inkling-mini` 只删除并重建 `inkling-mini-r0..3`。
`stop.sh` 也只删除这四个容器，不会停止 `deepseek-r0..3`。

全部停止仍使用显式的：

```bash
./deploy.sh stop
```

## 资源前置

- 四台节点都必须有相同镜像和完整模型；代码同步不会复制模型或镜像。
- 四台节点都必须能从统一 `/30` fabric 和 GID index 3 建立 ring-only
  NCCL communicator。
- 启动前要求各节点 `MemAvailable >= 20 GiB`。低于门限直接失败，不会自动
  停止 DeepSeek 腾资源。
- 首次启动需要编译 SGLang JIT kernel，缓存写入
  `/opt/runtime_assets/inkling-mini/cache`。

## 已知未验收项

- 四节点 ring-only NCCL、TP4/EP4、DSpark、decode graph、真实 chat 和
  DeepSeek 并行运行已通过。
- 当前 4 x 256K 配方已完成 prose c1/c2/c4 与 8K-128K cold prefill
  基线；后续优化应继续以同一 boot 的 A/B 和多次重复为准。
- `mem_fraction_static=0.72` 是当前 4 x 256K + 4096 prefill chunk 的
  最小安全值；0.718 在 4096/4096 下 KV pool 只有 `1,006,464` tokens，
  不足以覆盖 4 个完整 256K 请求。
