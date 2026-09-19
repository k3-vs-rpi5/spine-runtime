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

`scripts/build.sh` 为智算簇编译这两个二进制（它们的内核在簇上执行）：默认是
`-march=rv64imafdcv_zicsr_zifencei_zba_zbb_zbs_xsmtvdotii -mtune=spacemit-a100`，
即显式给出 A100 ISA 与厂商调度模型，而不用那把"打包式"的 `-mcpu` 开关。可用
`SPERT_A100_FLAGS` 覆盖。

选定它的矩阵如下，每个构建跑 5 次 `a100_bandwidth 64`（每次含 4 个变体，只有 4 个都
完成才计为成功一次）：

| 构建 | 完成次数 |
|---|---|
| 纯 `rv64gc` | 4/5 |
| 只用 `-mtune=spacemit-a100` | 3/5 |
| `-march` 不带向量 | 3/5 |
| `-march` 带向量 | 4/5 |
| `-march` 带向量与 `_xsmtvdotii` | 3/5 |
| `-mcpu=spacemit-a100` | **0/5** |

两个发现，以及一处对本文件早先说法的更正。更正：该项第一次测量（3 次）写成"这个标志
决定它们能否运行"；做到 5 次后真相更窄——只有打包式的 `-mcpu=spacemit-a100` 是确定性
失败，其余任何子集的表现都与纯构建无异。这也意味着工具**本身就在随机失败**：连纯构建
也大约五次里失败一次，这是工具或它对运行时的用法里的未解缺陷，而不是任何 ISA 选项的
问题，也正是默认值现在避开那把每次都失败的开关的原因。

内存天花板不受这些影响：纯构建与向量构建在纯流式上一致为 24.6 GiB/s，在解码路径自己
那条瓦片走法上接近 23 GiB/s，因此天花板是总线而不是指令组合。标量半字节解包在所有
构建里都读 2.7 GiB/s；要回答 4 bit 格式本身能否喂满总线，仍需手写向量化解包。

## 工作区登记

```xml
<project name="spine-runtime.git" path="projects/spine-runtime" />
```
