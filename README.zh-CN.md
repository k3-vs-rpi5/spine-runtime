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
- 自动化（`src/spine_runtime_project`、`scripts/build.sh`、`scripts/run.sh`）是下一个
  交付项，并已在 `project.yaml` 中标注为待完成。

## 工作区登记

```xml
<project name="spine-runtime.git" path="projects/spine-runtime" />
```
