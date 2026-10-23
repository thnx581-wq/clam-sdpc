# 本机验证记录

验证日期：2026-10-23（Asia/Shanghai）。这是精简后代码的验证记录，不代表所有切片或所有环境均已验证。

环境：Windows、Python 3.10.0、torch 2.14.0+cu130、timm 0.9.8、sdpc-win 3.0；GPU 为 NVIDIA GeForce RTX 5060 Ti。

`python -m unittest discover -s tests -v`：7 项通过。

- MPP 换算、原生层选择、重采样输出尺寸、旧 HDF5 属性读取。
- 空白切片返回空轮廓，不再因 findContours 无 hierarchy 崩溃。
- SDPC 左/上边界白色补齐，整块越界返回白色。
- 合成切片分割、坐标 HDF5、PNG mask、stitch 和重复运行跳过。
- 特征 HDF5/PT 行数、维度、坐标顺序和数值相等；损坏/NaN 与模型设置变化被拒绝。
- 源文件发现与排除名单。
- OpenSlide 原生层和单层 TIFF 虚拟金字塔、透明区域白色合成与分块读取上限。

一张本地真实 SDPC 的试跑结果：

- 读取 7 个金字塔层，level-0 MPP 为 0.2060639880952381。
- 默认 balanced preset 在 1.0 mpp、224×224、步长224 下生成 1,258 个坐标。
- 分割 PNG 和轮廓文件已生成。
- 从坐标中取 4 个 patch，使用本地 UNI v1 在 CUDA 上输出 `(4, 1024)`，HDF5/PT、坐标与有限数值检查通过。
- 相同 4 个 patch 使用本地 ResNet50 输出 `(4, 1024)`，同样检查通过。
- SDPC DLL 导入前后的进程工作目录一致。

`python pipeline.py --help` 与 `python -m compileall -q .` 通过。

本轮没有全量提取任何患者数据，没有运行 H-optimus 的完整 GPU 推理测试，也没有对真实 OpenSlide 切片做全流程试跑。H-optimus 保留原项目已使用的架构、严格权重加载和归一化配置。原始工程和既往结果未修改。验证用切片、权重、坐标、图像和特征位于被 Git 忽略的本地输出目录，不随源码上传。

依赖安装文档已整理，但没有重装现有环境或新建环境验证安装过程。
