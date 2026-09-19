[English](README.md)

# Spine-Runtime 项目

厂商针对智算核给出的编程模型，也是唯一有文档的提交入口。`libspert` 是预编译的；
API 与调度模型是公开的。

## 固定来源

| 字段 | 取值 |
|---|---|
| 上游 | `https://github.com/spacemit-com/spine-runtime.git` |
| ref | `main` |
| 提交 | `b3ff5182de116613b6db72b0f43064554ac3d42d` |
| 许可证 | MIT |
| 检出 | `src/spine-runtime` |
| SDK 包 | `spine-runtime.riscv64.0.6.2.tar.gz`，`libspert.so.0.6.2` |

模型是 Tile-SPMD：`Stream` 持有后端授予的计算核，`Grid` 把工作切成 tile，每个 tile
从 `Context` 读取自己的坐标。`backend_info()` 报告实际获得的核心数、共享缓冲大小、
向量长度与架构 id。`spacemit-k3` 后端面向智算核 8-15，每核 384 KiB 缓冲；
`spacemit-k3-x100` 是 X100 核上的兼容模式，`generic/qemu` 仅供功能验证。

## 状态

- 文档仓库与 SDK 身份已同步；SDK 作为密封输入留在 Git 之外，其摘要写在
  `sources.lock.yaml` 中。
- 板上尚未安装 `spacemit-runtime`，因此第一个 A100 程序仍需该包或预编译 SDK。
- 自动化已就位：`scripts/build.sh` 用 SDK 与板卡的 `spine_tcm` 头文件交叉编译
  `src/spine_runtime_project/a100_probe.cc`，`scripts/run.sh` 负责交付、执行并把输出留作证据。
- 板上实测：默认后端报告 `arch_id 0xa064`（A100）并授予 8 个核（id 8-15）；900000 次
  启动的窗口以厂商侧为证据——运行时对这批工作的注册、授予提交方 pid 的 TCM 块——再加上
  该内核自身尺寸与启动次数所蕴含的吞吐算术。逐 CPU 计数器与 `/proc/stat` 会被记录，
  但在此板上不是 A100 证据：2026-09-18 实测智算核的工作不会推动它们。

## A100 ISA 标志与我们提交的内核

`scripts/build.sh` 为智算簇编译这两个二进制（`-mcpu=spacemit-a100`，可用
`SPERT_A100_FLAGS` 覆盖），因为它们的内核在簇上执行。2026-09-19 实测：这个标志决定
它们能否运行：

| 构建 | `a100_bandwidth 64` 完成运行的次数 |
|---|---|
| 纯 `rv64gc` | 3/3 |
| `-mcpu=spacemit-a100` | 0/3 |
| `-mcpu=spacemit-a100 -fno-tree-vectorize` | 0/3 |

关闭向量化后同样崩溃，所以不是自动向量化的锅；而简单的探针内核在两种构建下都能运行，
所以也不是该标志打挂所有内核。在弄清之前，下面的带宽数字都是标量代码的数字——但它们
依然有价值：标量流式内核在 1.5 GiB 工作集下已经跑到 **24,627 MiB/s**，与 X100 侧探针
给出的天花板一致。因此无论指令组合如何，内存天花板都成立；串行解码的基准就是它自己
那条 65,792 字节瓦片走法（22,658 MiB/s，比纯流式低 8 %）。

## 工作区登记

```xml
<project name="spine-runtime.git" path="projects/spine-runtime" />
```
