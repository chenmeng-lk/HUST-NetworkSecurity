# 信息隐藏算法测试指引

LSB算法实现彩图中隐写黑白水印、音频中隐写字符串，DCT算法实现灰度图中隐写字符串

------

## 文件结构

------U202215494-廖奎-信息隐藏研究

-------------reference

-------------NetworkSecurity

--------------------------code

---------------------------------------*.py

---------------------------------------requirements.txt

--------------------------resources

---------------------------------------LSB

---------------------------------------DCT

reference存放的是是参考文献

NetworkSecurity中存放的是源代码和测试图片资源

code中包含运行代码*.py和存储库依赖包的安装文件requirements.txt

resources中存放代码运行样例资源和运行中生成的文件

## 环境

PowerShell终端运行python

## 运行方式

把**NetworkSecurity**文件复制到**D盘直接目录**下（D:/NetworkSecurity），进入code目录，空白处右击，点击“在终端中打开”，输入以下命令检查 Python 版本

```powershell
python --version
```

若未安装python，查看https://blog.csdn.net/qq_53280175/article/details/121107748安装教程

已安装好python后，创建虚拟环境安装必要支持库，输入以下命令在项目目录下创建虚拟环境：

```powershell
python -m venv venv
```

输入以下命令激活虚拟环境

```powershell
.\venv\Scripts\Activate.ps1
```

若激活失败，运行以下命令检查当前的执行策略

```powershell
Get-ExecutionPolicy
```

如果输出结果是 Restricted，说明禁止执行任何脚本。需要将其更改为 RemoteSigned以允许运行本地的脚本。通过以下命令临时更改 PowerShell 执行策略

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

更改策略后，再次尝试激活虚拟环境

```powershell
.\venv\Scripts\Activate.ps1
```

激活后，命令行提示符会变成 (venv)，表示虚拟环境已激活。安装依赖库

```powershell
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple setuptools decorator imageio colorama pillow proglog numpy opencv-python python-dotenv
```

安装完成后就可以开始运行Python 代码了！

## 运行程序

输入以下命令利用LSB将输入的字符串转换成黑白水印，隐写到彩色图片中

```powershell
python LSB_image.py
```

将隐写后的彩色图片添加白色遮盖，再提取水印

```powershell
python LSB_image_cover.py
```

将隐写后的彩色图片压缩，再提取水印

```powershell
python LSB_image_compress.py
```

输入以下命令利用LSB将输入的字符串隐写到音频中

```powershell
python LSB_audio.py
```

处理隐写后的音频，再提取字符串

```powershell
python LSB_audio_comp_trim.py
```

输入以下命令利用DCT将输入的字符串隐写到灰度图片中

```powershell
python DCT.py
```

处理隐写后的灰度图片，再提取字符串

```powershell
python DCT_robustness.py
```

想要退出时，通过以下命令退出虚拟环境

```powershell
deactivate
```

## 样例资源

#### LSB

LSB.png  原始彩图

LSB_embedded.png  嵌入黑白水印生成的载体彩图

LSB_create_watermark.png  输入字符串生成的水印

LSB_get_watermark.png  从载体彩图提取出的黑白水印

LSB_cover.png  遮盖载体彩图

LSB_cover_watermark.png  从遮盖后的载体彩图提取出的水印

LSB_compressed_jpg  压缩载体彩图

LSB_compressed_watermark.png    从压缩后的载体彩图提取出的水印 

LSB.wav  原始音频

LSB_embedded.wav   嵌入字符串后生成的载体音频

LSB_compressed.wav   压缩载体音频

LSB_trimmed.wav   裁剪载体音频

#### DCT

DCT.bmp  原始灰度图

DCT_embedded.bmp   嵌入字符串生成的载体图片  

DCT_cover.bmp  添加遮挡的载体图片

DCT_embedded_compressed.jpg   压缩后的载体图片

DCT_embedded_rotated.bmp   旋转后的载体图片

DCT_wrong.bmp   错误样例，有明显黑影，可见报告第22页描述

