# CLAM SDPC preprocessing

精简的 CLAM 病理切片预处理工具：组织前景分割 → 物理分辨率 patch 坐标 → UNI / H-optimus / ResNet50 特征。Windows 下使用 `sdpc-win==3.0` 读取 **`.sdpc`**；其他格式走 OpenSlide。`.sdpi` 暂无读取器，不要直接改后缀当作支持。

本项目源自 [Mahmood Lab CLAM](https://github.com/mahmoodlab/CLAM)。保留 GPL-3.0 和 [来源/改动说明](NOTICE.md)。训练、交叉验证、分类评估、预测热图、旧的 patch 图像存储以及患者专用脚本已移除。代码不包含切片、患者信息、模型权重或解码 DLL。

## 安装（Windows / PowerShell）

建议创建独立环境。已使用本机 Python 3.10、timm 0.9.8 验证。

```powershell
conda create -n clam_sdpc python=3.10 pip -y
conda activate clam_sdpc
python -m pip install torch torchvision
python -m pip install -r requirements.txt
```

GPU 版 torch/torchvision 的配对与 CUDA wheel 源，请按 [PyTorch 安装页面](https://pytorch.org/get-started/locally/)选择。上面的普通安装命令不保证获得适配你显卡的 CUDA 版本。若使用现有 `clam_latest` 环境，可直接运行，无需重建。`env.yml` 仅安装读取/图像依赖，torch 和 torchvision 仍按设备选择；不再需要 smooth-topk、tensorboardX、scikit-learn、pandas 或 matplotlib。

SDPC Windows DLL 由 pip 包安装。当前未接入 Linux SDPC 解码。普通 OpenSlide 文件依赖自身可用的 MPP 元数据；缺失、非法或明显各向异性时报告错误，不猜测倍率。

## 常用命令

先进入本仓库目录。下列盘符和路径仅为示例。

**仅分割，先看 PNG：**

```powershell
python pipeline.py --source "E:\slides" --output "D:\outputs\case_1mpp" --stage segment --limit 3
```

**分割和坐标划分，1.0 mpp、224×224、无重叠：**

```powershell
python pipeline.py --source "E:\slides" --output "D:\outputs\case_1mpp" --stage preprocess --target-mpp 1.0 --patch-size 224 --stitch
```

**使用已有坐标提取 UNI 特征：**

```powershell
python pipeline.py --source "E:\slides" --output "D:\outputs\case_1mpp" --stage features --target-mpp 1.0 --patch-size 224 --model uni_v1 --checkpoint "D:\weights\uni_v1.bin" --device cuda --batch-size 64
```

**一条命令处理 0.5 mpp 的三个样本：**

```powershell
python pipeline.py --source "E:\slides" --output "D:\outputs\case_05mpp" --target-mpp 0.5 --patch-size 224 --limit 3 --model uni_v1 --checkpoint "D:\weights\uni_v1.bin" --device cuda --stitch
```

去掉 `--limit 3` 即处理源目录所有支持的切片（不递归）。指定文件用 `--slides sample_a.sdpc sample_b.sdpc`。排除名单用 `--exclude exclude.txt`，UTF-8 文本每行一个完整文件名或不带后缀的文件名；不接受含患者信息的旧 JSON 清单。不要把真实名单提交到 GitHub。

H-optimus 用 `--model h_optimus_0 --checkpoint "D:\weights\H_optimus\pytorch_model.bin" --amp`。实现复用原工作流的 timm DINOv2 giant/reg4/224 架构，本地严格加载权重，采用该模型专用均值/标准差；不需要运行时 Hugging Face 登录。输入固定缩放为 224×224，输出 1536 维。UNI v1 输出 1024 维。ResNet 使用 `--model resnet50_trunc`，输出 1024 维；可传本地 ImageNet checkpoint，否则 timm 首次下载权重。大型模型显存不足时降低 `--batch-size`。

Windows 默认 `--workers 0`，避免此前遇到的系统提交内存/页面文件不足。内存允许时可试 `--workers 1` 或 `2`；每个进程独立打开切片，DLL 句柄不会跨进程共享。`--amp` 在 CUDA 下开启 float16 混合精度；它会影响数值，已有特征目录会核对该设置。

## 原有坐标和输出兼容

已有坐标 HDF5 可用 `--stage features --coords-dir "D:\old_output\patches" --output "D:\new_features"` 复用，必须传入与坐标属性相符的 `--target-mpp`、`--patch-size`。读取旧 CLAM HDF5 时，dataset 仍支持仅有 `patch_level/patch_size` 的旧属性；通用 CLI 的 MPP 验证要求物理分辨率属性，因此没有 `target_mpp` 的旧文件不能在此冒充已确认的物理分辨率。

```text
output/
  preprocess_config.json       源目录、分割和 patch 设置
  run_status.csv               本次运行逐切片状态；失败后继续下一张
  logs/pipeline.log            累积错误与状态
  masks/<slide>.png            轮廓叠加图（非二值 mask）
  contours/<slide>.pkl          level-0 组织轮廓及孔洞，仅加载可信的本地文件
  patches/<slide>.h5           coords: (N, 2)
  stitches/<slide>.png         可选 patch 拼接预览
  features/uni_v1/h5_files/    features: (N, 1024)，coords: (N, 2)
  features/uni_v1/pt_files/    Tensor: (N, 1024)
```

H-optimus、ResNet 使用各自模型名的子目录。HDF5 和 PT 特征逐行一致，coords 是 **level-0 左上角坐标，顺序为 (x, y)**。恢复运行时核对分割配置、MPP、patch 尺寸、模型信息、坐标顺序和有限数值；不完整特征重算当前切片。每张成功后才提交最终特征文件；`.tmp` 和 `.incomplete` 是未完成中间产物。不要同时启动两个进程写同一个输出目录。

重跑相同命令可续跑。修改分割参数、MPP、patch 步长或模型权重时使用新的输出目录，便于保存旧结果。源切片应保持不变。`run_status.csv` 记录当前这次执行；累积记录在日志中。某张失败会继续后续切片，最终退出码为 1；全部成功为 0。

## MPP 与分割参数

MPP 表示微米/像素，不能单凭“10×/20×”确认。SDPC 使用文件头 `ruler` 作为 level-0 MPP。给定输出 patch 大小 P、目标 MPP T 和原生 MPP B，level-0 理想边长为 `P*T/B`；先选择最接近的金字塔层，按相应视野读取，再重采样到 P×P。

整数坐标和读取尺寸沿用原工作流的取整方法。`target_mpp` 是请求的名义 MPP：level-0 边长与理想值之差至多半个 level-0 像素，原生读取边长有至多半个所选层像素的取整误差；不应将它解释成数学上无误差的精确 MPP。1.0 mpp 的 224 patch 约覆盖 224×224 µm；0.5 mpp 则约覆盖 112×112 µm。

坐标属性保留 `base_mpp`、`target_mpp`、`source_level`、`source_read_size`、`output_patch_size`、`native_level_mpp`、`level0_patch_size`、`level0_step_size`。读取边界以白色填充；DLL 导入后的工作目录会恢复。单层 TIFF 提供分块读取的虚拟金字塔，避免整张高分辨率切片一次性进入内存。

默认 [presets/balanced.json](presets/balanced.json) 延续先前 balanced_v2：HSV 饱和度阈值16、median7、close3、`a_t=100/a_h=8/max_n_holes=32`，patch 中心位于组织轮廓且不在孔洞才纳入。绿色为组织外轮廓，蓝色为孔洞。它不是逐 patch 的“空白≤30%”过滤；本精简版未包含另行运行的边界空白筛除脚本。

复制 preset JSON 后通过 `--preset your_preset.json` 调整：提高 `sthresh` 更严格，降低则保留浅染组织；增加 `a_t` 去除较小区域；减小 `a_h` 可保留更多孔洞；`four_pt_hard` 比 `center` 更严格。`a_t/a_h` 为 CLAM 按分割层缩放的参考面积单位，不是百分比。调参先用 `--stage segment --limit 3` 和新输出目录确认组织保留情况。

## 验证与发布

```powershell
python -m unittest discover -s tests -v
python pipeline.py --help
```

测试覆盖合成切片的 MPP 换算、白边补齐、空组织、原属性兼容、分割→坐标→stitch、特征行对齐、恢复和损坏检测；不替代对真实切片的视觉验收。具体本机验证见 [docs/VALIDATION.md](docs/VALIDATION.md)。

上传方法见 [docs/GITHUB.md](docs/GITHUB.md)。GitHub 上传称为 **push**；pull 是将远端更新取到本地。
