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
- 板上实测：默认后端报告 `arch_id 0xa064`（A100）、获批 8 个核，在 900000 次
  启动的窗口内 `/proc/stat` 显示智算核忙碌 59.9 %，与 per-CPU 计数器一致。

## 工作区登记

```xml
<project name="spine-runtime.git" path="projects/spine-runtime" />
```
