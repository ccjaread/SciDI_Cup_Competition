## 概述
### 主要思路和文件组成
#### 赛题链接
![jpg](https://tianchi-public.oss-cn-hangzhou.aliyuncs.com/public/files/forum/159114959758937451591149597964.jpeg)

#### 主要思路
1. 首先使用规则匹配出可能有天文事件的片段，并生成50*50像素的图片（之所以是50*50是为了人看方便，本来计划是LeNet一样的28*28）
2. 再应用一个类似LeNet的小规模神经网络识别事件
3. 按事件概率大小排序，筛选出候选样本
4. 按需通过X阶（默认X=5）自回归系数>0.8，或事件是否跨越两段不同时间测得的数据，作为筛选条件进一步筛选

#### 效果
- 亚军
#### 文件组成

```
project
    |--README.md 本说明文档
    |--user_data
        |-- train_data
            |-- Train92 由训练样本生成的候选数据，文件夹名称编号主要用于与不同版本代码保持一致
                |-- 0 放置由训练的18个样本通过脚本batch_img_detect.py 生成的非事件图片
                |-- 1 放置由训练的18个样本通过脚本batch_img_detect.py 生成的flare star图片
                |-- 2 放置由训练的18个样本通过脚本batch_img_detect.py 生成的microlening图片
        |-- trained_model 放置了几个不同训练程度的LeNet模型，可供测试或继续训练，默认调用09122版本
        |-- finded_result 放置了需要检测的80多万序列通过规则匹配产生的候选50*50的图片
        |-- starid_path_dict.pickle 初赛的starid 到对应数据文档路径的映射字典
        |-- starid_path_dict2.pickle 复赛的starid 到对应数据文档路径的映射字典
            #生成starid 到对应数据文档路径的映射字典
            # 
            #files_lst=glob.glob('./ref*')
            #files_lst为所有文档的路径
            # starid_path_dict={}
            # for file_path in files_lst:
            #     starid_path_dict[file_path.split('/')[-1]]=file_path

            # with open('./starid_path_dict.pickle','wb') as f:
            #     pickle.dump(starid_path_dict,f)

        
    |--prediction_result
    |--code 本目录下给了代码的py和jupyter notebook 版本，推荐使用jupyter notebook 版本，思路较清晰
        |-- step01_batch_img_detect.py 对应思路步骤1，通过抽象数据序列并通过规则筛选出候选事件图片，需修改路径后使用
        |-- step02_predit_lenet.py 对应步骤思路3、4，即从user_data/finded_result/中读取并判别图片，加上筛选条件后得到(py版，方便运行，目前默认生成20200912提交的结果，只包含复赛的数据)
        |-- predit_lenet.ipynb 对应步骤思路3、4，即从user_data/finded_result/中读取并判别图片，加上筛选条件后得到(notebook版方便修改)
        |-- predit_lenet.md predit_lenet.ipynb的说明，方便转成py，但ipynb文件中的思路更明显
        |-- train_lenet.ipynb 对应思路步骤2，训练LeNet产生模型的过程，主要通过读取主办方提供的事件模板加了随机干扰生成了许多人造样本
        |--AstroSet 下载的数据文档，因为程序使用相对路径，所以这个最好摆在这边或者重新制作starid_path_dict文件
            |--021-*******等
            |--022-*******等
```

### 运行环境
#### 计算机配置
- 目前的笔记本，内存大于4G的应该都能运行，使用3个进程并行实际占用内存在3G左右，使用3个进程完成总体80万数据以下列配置需要处理约8小时

```
处理器名称    QuadCore Intel Core i3-8100T, 3100 MHz (31 x 100)
主板名称     Lenovo ThinkCentre M720q
主板芯片组   Intel Cannon Point B360, Intel Coffee Lake-S
系统内存     8064 MB  (DDR4 SDRAM)

显示设备:
显示适配器  Intel(R) UHD Graphics 630  (1 GB)
显示适配器  Intel(R) UHD Graphics 630  (1 GB)
显示适配器  Intel(R) UHD Graphics 630  (1 GB)
3D 加速器  Intel UHD Graphics 630

```

#### 第三方 Python 包
- 目前使用的如下，除TensorFlow版本敏感外，其他均为通用包

```
pandas                        0.25.3 
numpy                         1.17.4+mkl
matplotlib                    3.2.0rc2
scikit-learn                  0.22
scipy                         1.4.1 
multiprocess                  0.70.9
tensorflow-cpu                2.1.0rc1
Keras                         2.3.1 是tensorflow自带的
```




