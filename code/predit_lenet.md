```python
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import matplotlib.image as mpimg

scaler = preprocessing.MinMaxScaler((0,1))

import glob
import re
import pickle
```


```python
## 定义需要处理的函数
def img_to_array(imageFile):
    img = mpimg.imread(imageFile).astype(np.float)
    img=img[:,:,0]
    img=1-img
    return img.reshape((50,50,1))

def find_name(x):
    return re.search(r'(ref_.*)\+\d',x).group(1)

def find_pos(path):
    return re.search(r'ref_.*\+(\d+-\d+)\.p',path).group(1)

def get_real_pos(path):
    path=path['path']
#     print(type(path))
    try:
        starid=find_name(path)
        rel_pos=find_pos(path)
        l_pos_idx,r_pos_idx=rel_pos.split('-')
        
        #按格式整理后的路径修改
        path=starid_path_dict.get(starid)
        #次处按相对路径做了修改，初始代码和数据是放在一个文件夹了，现在多了一层/code/将.改成..
        path=path.replace(r'.',r'..')
        
        data=pd.read_table(path,sep=u' ',names=['time','data','dev'])
        data.sort_values(by='time',ascending=True,inplace=True)
        return data.time[eval(l_pos_idx)],data.time[eval(r_pos_idx)]
    except:
        return 'NN','NN'

def get_rel_pos(path):
    path=path['path']
#     print(type(path))
    try:
        starid=find_name(path)
        rel_pos=find_pos(path)
        l_pos_idx,r_pos_idx=rel_pos.split('-')
        return eval(l_pos_idx),eval(r_pos_idx)
    except:
        return 'NN','NN'

def get_breakpoints_apply(df):
    
    #按格式整理后的路径修改
    path=starid_path_dict.get(df['starid'])
    #次处按相对路径做了修改，初始代码和数据是放在一个文件夹了，现在多了一层/code/将.改成..
    path=path.replace(r'.',r'..')
    
    data=pd.read_table(path,sep=u' ',names=['time','data','dev'])
    data.sort_values(by='time',ascending=True,inplace=True)
    breakpoints=get_breakpoints(data)
    return breakpoints

def get_breakpoints(data):
    diff=data.time.diff()
    diff.fillna(method='bfill',inplace=True)
    breakpoints,=np.where(diff.values>diff.mean()*4)
    breakpoints=np.append(breakpoints,len(data))
    breakpoints=np.append(0,breakpoints)
    return breakpoints

def show_breakpoints(data,breakpoints):
    diff=data.time.diff()
    diff.fillna(method='bfill',inplace=True)
    fig, ax1 = plt.subplots(figsize=(5,5))
    ax2 = ax1.twinx()
    ax1.plot(range(len(data)),data.time,c='b')
    ax2.scatter(range(len(data)),diff,c='r')
    ax2.vlines(breakpoints,0,diff.max(),linestyles='dashed')
    ax1.set_xlabel("index")
    ax1.set_ylabel("time")
    ax2.set_ylabel("time_diff")
    plt.show()

def get_whole_data(insp_path):
    
    ##按格式整理后的路径修改
    #次处按相对路径做了修改，初始代码和数据是放在一个文件夹了，现在多了一层/code/将.改成..
    insp_path=insp_path.replace(r'.',r'..')
    
    data=pd.read_table(insp_path,sep=u' ',names=['time','data','dev'])
    data.sort_values(by='time',ascending=True,inplace=True)
    breakpoints=get_breakpoints(data)
    # show_breakpoints(data,breakpoints)
    whole_data=np.array([0])
    #     whole_dev=np.array([0])
    for i in range(len(breakpoints)-1):
        new_data=data.loc[breakpoints[i]:breakpoints[i+1]-1,'data'].values
        gap=np.median(whole_data)-np.median(new_data)
        new_data=new_data+gap
        whole_data=np.concatenate((whole_data,new_data))
    return whole_data

def autocorrelation(x,lags):#计算lags阶以内的自相关系数，返回lags个值，分别计算序列均值，标准差
    n = len(x)
    x = np.array(x)
    result = [np.correlate(x[i:]-x[i:].mean(),x[:n-i]-x[:n-i].mean())[0]/(x[i:].std()*x[:n-i].std()*(n-i)) for i in range(1,lags+1)]
    return result[-1]

def autocorrelation_diff(x,lags):#计算lags阶以内的自相关系数，返回lags个值，分别计算序列均值，标准差
    n = len(x)
    x = np.array(x)
    result = [np.correlate(x[i:]-x[i:].mean(),x[:n-i]-x[:n-i].mean())[0]/(x[i:].std()*x[:n-i].std()*(n-i)) for i in range(1,lags+1)]
    result = np.array(result)
    diff_=result[:-1]-result[1:]
    return diff_.mean()

def get_auto_corr(starid,lag=5):
    whole_data=get_whole_data(starid_path_dict.get(starid))
    auto_corr=autocorrelation(-whole_data,lag)
    return auto_corr
```


```python
#生成starid 到对应数据文档路径的映射字典
# with open('./file_lst.pickle','rb') as f:
#     files_lst=pickle.load(f)

# starid_path_dict={}
# for file_path in files_lst:
#     starid_path_dict[file_path.split('/')[-1]]=file_path

# with open('./starid_path_dict.pickle','wb') as f:
#     pickle.dump(starid_path_dict,f)
```


```python
#starid_path_dict 初赛的starid 到对应数据文档路径的映射字典
#starid_path_dict2 复赛的starid 到对应数据文档路径的映射字典
with open('../user_data/starid_path_dict2.pickle','rb') as f:
    starid_path_dict=pickle.load(f)
```


```python
path_pred=r'../user_data/finded_result/*.png'
```


```python
pool_pred=glob.glob(path_pred)
```


```python
print('读取到候选天体事件{}个'.format(len(pool_pred)))
```

    读取到候选天体事件21608个
    


```python
#as files are still be generated record the current files
# with open('./pool_pred_20200729.pickle','wb') as f:
#     pickle.dump(pool_pred,f)
```


```python
#读取训练的模型，模型迭代好几版了，各有优缺点，默认20200909版，注意tensorflow版本，亦可重新训练下
from tensorflow.keras.models import load_model
model = load_model("../user_data/trained_model/LeNet_20200909_img_50_50.h5")

model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 46, 46, 6)         156       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 23, 23, 6)         0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 20, 20, 16)        1552      
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 10, 10, 16)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 8, 8, 32)          4640      
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 4, 4, 32)          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 512)               0         
    _________________________________________________________________
    dense (Dense)                (None, 120)               61560     
    _________________________________________________________________
    dropout (Dropout)            (None, 120)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 84)                10164     
    _________________________________________________________________
    dense_2 (Dense)              (None, 3)                 255       
    =================================================================
    Total params: 78,327
    Trainable params: 78,327
    Non-trainable params: 0
    _________________________________________________________________
    


```python
pool_pred_starids=set([name for name in map(find_name,pool_pred)])
```


```python
x_size=len(pool_pred)
x = np.zeros((x_size, 50,50,1))
#train sample only
for i in range(x_size):
    x[i] = img_to_array(pool_pred[i])
```


```python
pred_y=model.predict(x)

y=np.apply_along_axis(np.argmax,1,pred_y)

score=np.apply_along_axis(np.max,1,pred_y)
```


```python
predicts=pd.DataFrame({'starid':[name for name in map(find_name,pool_pred)],\
                       'event':y,\
                       'score':score,\
                       'path':pool_pred})
```


```python
predicts.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>starid</th>
      <th>event</th>
      <th>score</th>
      <th>path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ref_021_24960425-G0014_628217_42591</td>
      <td>0</td>
      <td>1.0</td>
      <td>../user_data/finded_result\ref_021_24960425-G0...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ref_021_24960425-G0014_628618_19794</td>
      <td>0</td>
      <td>1.0</td>
      <td>../user_data/finded_result\ref_021_24960425-G0...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ref_021_24960425-G0014_628619_19671</td>
      <td>0</td>
      <td>1.0</td>
      <td>../user_data/finded_result\ref_021_24960425-G0...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ref_021_24960425-G0014_628668_18312</td>
      <td>0</td>
      <td>1.0</td>
      <td>../user_data/finded_result\ref_021_24960425-G0...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ref_021_24960425-G0014_628684_16349</td>
      <td>0</td>
      <td>1.0</td>
      <td>../user_data/finded_result\ref_021_24960425-G0...</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(predicts.loc[predicts.event!=0].sort_values(by='score',ascending=False))
```




    155




```python
predicts.loc[predicts.event!=0].sort_values(by='score',ascending=False).head(50).starid.to_list()
```




    ['ref_021_24960425-G0014_641731_14386',
     'ref_044_12870595-G0013_448543_22840',
     'ref_022_14110425-G0014_370643_5999',
     'ref_043_12010765-G0013_496916_197051',
     'ref_044_00940255-G0013_1273778_7291',
     'ref_043_04700255-G0013_29528_21239',
     'ref_043_00940255-G0013_1265997_24761',
     'ref_043_04700255-G0013_28874_19479',
     'ref_043_12010765-G0013_474914_29466',
     'ref_041_16810765-G0013_510950_21479',
     'ref_044_14110425-G0013_410983_22399',
     'ref_043_04700255-G0013_28382_6790',
     'ref_044_18590595-G0013_382080_13703',
     'ref_044_04700255-G0013_45239_27087',
     'ref_034_21640255-G0013_569111_13375',
     'ref_043_04700255-G0013_28122_34945',
     'ref_043_06300085-G0013_1457022_9648',
     'ref_044_14110425-G0013_326441_15380',
     'ref_044_12010765-G0013_500055_12380',
     'ref_044_34820255-G0013_1287904_7965',
     'ref_043_06300085-G0013_1491366_16831',
     'ref_044_04500085-G0013_1133981_14981',
     'ref_043_06300085-G0013_1469618_32402',
     'ref_041_11940425-G0013_434054_36613',
     'ref_044_12230255-G0013_1441255_28735',
     'ref_043_12010765-G0013_474910_90832',
     'ref_044_14110425-G0013_324602_9737',
     'ref_044_14110425-G0013_321481_15782',
     'ref_044_34820255-G0013_1288349_20418',
     'ref_044_14110425-G0013_321183_19981',
     'ref_044_14110425-G0013_325266_7324',
     'ref_044_14110425-G0013_415747_25237',
     'ref_044_14110425-G0013_323735_10140',
     'ref_044_00940255-G0013_1273410_7321',
     'ref_044_14110425-G0013_318062_6704',
     'ref_044_12230255-G0013_304293_4238',
     'ref_044_14110425-G0013_327137_6672',
     'ref_044_14110425-G0013_321195_22023',
     'ref_043_00940255-G0013_1267593_23600',
     'ref_044_12010765-G0013_501847_90690',
     'ref_043_04700255-G0013_27969_13147',
     'ref_044_14110425-G0013_414293_19782',
     'ref_044_15730595-G0013_376489_10570',
     'ref_043_12010765-G0013_496916_86866',
     'ref_044_14110425-G0013_324086_6754',
     'ref_044_14110425-G0013_320949_13736',
     'ref_044_14110425-G0013_324247_12940',
     'ref_043_04700255-G0013_27711_11506',
     'ref_044_14110425-G0013_412983_28297',
     'ref_044_04500085-G0013_1137840_8821']




```python
# predicts.loc[predicts.event!=0].sort_values(by='score',ascending=False).to_csv('res_20200909.csv')
```


```python
final_predicts=predicts.loc[predicts.event!=0].sort_values(by='score',ascending=False).copy()
```


```python
final_predicts.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>starid</th>
      <th>event</th>
      <th>score</th>
      <th>path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>166</th>
      <td>ref_021_24960425-G0014_641731_14386</td>
      <td>1</td>
      <td>1.000000</td>
      <td>../user_data/finded_result\ref_021_24960425-G0...</td>
    </tr>
    <tr>
      <th>15522</th>
      <td>ref_044_12870595-G0013_448543_22840</td>
      <td>1</td>
      <td>1.000000</td>
      <td>../user_data/finded_result\ref_044_12870595-G0...</td>
    </tr>
    <tr>
      <th>706</th>
      <td>ref_022_14110425-G0014_370643_5999</td>
      <td>1</td>
      <td>1.000000</td>
      <td>../user_data/finded_result\ref_022_14110425-G0...</td>
    </tr>
    <tr>
      <th>8124</th>
      <td>ref_043_12010765-G0013_496916_197051</td>
      <td>2</td>
      <td>1.000000</td>
      <td>../user_data/finded_result\ref_043_12010765-G0...</td>
    </tr>
    <tr>
      <th>9612</th>
      <td>ref_044_00940255-G0013_1273778_7291</td>
      <td>2</td>
      <td>0.999999</td>
      <td>../user_data/finded_result\ref_044_00940255-G0...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#get real time
final_predicts['l_pos']='N'
final_predicts['r_pos']='N'
final_predicts[['l_pos','r_pos']]=final_predicts.apply(get_real_pos,axis=1,result_type='expand')

#get rel idx
final_predicts['l_pos_rel']='N'
final_predicts['r_pos_rel']='N'
final_predicts[['l_pos_rel','r_pos_rel']]=final_predicts.apply(get_rel_pos,axis=1,result_type='expand')

#计算5阶自回归系数，以排除一部分周期性的脉冲星，看情况使用作为筛选条件
final_predicts['auto_corr']=final_predicts['starid'].apply(get_auto_corr)

#计算事件是否由两段时间观察得到的结果
final_predicts['breakpoints']=final_predicts.apply(get_breakpoints_apply,axis=1,result_type='reduce')

def overlap_break_p(df):
    bps=df['breakpoints']
    l=df['l_pos_rel']
    r=df['r_pos_rel']
    m_pos=(l+r)/2
    gap=r-l
    for bp in bps:
        if bp>=m_pos-0.015*gap and bp<=m_pos+0.015*gap:
            return 'Y'
    return 'N'

final_predicts['is_overlap_bp']=final_predicts.apply(overlap_break_p,axis=1)

#重新命名
final_predicts['event']=final_predicts['event'].replace({2:'microlensing',1:'flare star'})
```


```python
## 提交的最好成绩是这条
final_predicts.loc[(final_predicts.auto_corr<0.8) & (final_predicts.score>0.98),\
                   ['starid','l_pos', 'r_pos','event']].head(50)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>starid</th>
      <th>l_pos</th>
      <th>r_pos</th>
      <th>event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>166</th>
      <td>ref_021_24960425-G0014_641731_14386</td>
      <td>2.458606e+06</td>
      <td>2.458606e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>15522</th>
      <td>ref_044_12870595-G0013_448543_22840</td>
      <td>2.458492e+06</td>
      <td>2.458502e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>8124</th>
      <td>ref_043_12010765-G0013_496916_197051</td>
      <td>2.458519e+06</td>
      <td>2.458526e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>9612</th>
      <td>ref_044_00940255-G0013_1273778_7291</td>
      <td>2.458501e+06</td>
      <td>2.458504e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>6063</th>
      <td>ref_043_04700255-G0013_29528_21239</td>
      <td>2.458512e+06</td>
      <td>2.458519e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>4475</th>
      <td>ref_043_00940255-G0013_1265997_24761</td>
      <td>2.458486e+06</td>
      <td>2.458486e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>5902</th>
      <td>ref_043_04700255-G0013_28874_19479</td>
      <td>2.458509e+06</td>
      <td>2.458512e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>7604</th>
      <td>ref_043_12010765-G0013_474914_29466</td>
      <td>2.458519e+06</td>
      <td>2.458526e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>3194</th>
      <td>ref_041_16810765-G0013_510950_21479</td>
      <td>2.458505e+06</td>
      <td>2.458505e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>19032</th>
      <td>ref_044_14110425-G0013_410983_22399</td>
      <td>2.458496e+06</td>
      <td>2.458499e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>5839</th>
      <td>ref_043_04700255-G0013_28382_6790</td>
      <td>2.458508e+06</td>
      <td>2.458509e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>20298</th>
      <td>ref_044_18590595-G0013_382080_13703</td>
      <td>2.458488e+06</td>
      <td>2.458490e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>13013</th>
      <td>ref_044_04700255-G0013_45239_27087</td>
      <td>2.458488e+06</td>
      <td>2.458496e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>5786</th>
      <td>ref_043_04700255-G0013_28122_34945</td>
      <td>2.458488e+06</td>
      <td>2.458490e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>6212</th>
      <td>ref_043_06300085-G0013_1457022_9648</td>
      <td>2.458516e+06</td>
      <td>2.458518e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>18204</th>
      <td>ref_044_14110425-G0013_326441_15380</td>
      <td>2.458496e+06</td>
      <td>2.458499e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>14009</th>
      <td>ref_044_12010765-G0013_500055_12380</td>
      <td>2.458468e+06</td>
      <td>2.458468e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>21065</th>
      <td>ref_044_34820255-G0013_1287904_7965</td>
      <td>2.458499e+06</td>
      <td>2.458499e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>6680</th>
      <td>ref_043_06300085-G0013_1491366_16831</td>
      <td>2.458516e+06</td>
      <td>2.458518e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>10275</th>
      <td>ref_044_04500085-G0013_1133981_14981</td>
      <td>2.458467e+06</td>
      <td>2.458469e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>6485</th>
      <td>ref_043_06300085-G0013_1469618_32402</td>
      <td>2.458510e+06</td>
      <td>2.458518e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>15291</th>
      <td>ref_044_12230255-G0013_1441255_28735</td>
      <td>2.458510e+06</td>
      <td>2.458523e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>7597</th>
      <td>ref_043_12010765-G0013_474910_90832</td>
      <td>2.458519e+06</td>
      <td>2.458526e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>17540</th>
      <td>ref_044_14110425-G0013_324602_9737</td>
      <td>2.458496e+06</td>
      <td>2.458499e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>16310</th>
      <td>ref_044_14110425-G0013_321481_15782</td>
      <td>2.458496e+06</td>
      <td>2.458499e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>21151</th>
      <td>ref_044_34820255-G0013_1288349_20418</td>
      <td>2.458499e+06</td>
      <td>2.458499e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>16151</th>
      <td>ref_044_14110425-G0013_321183_19981</td>
      <td>2.458496e+06</td>
      <td>2.458499e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>17677</th>
      <td>ref_044_14110425-G0013_325266_7324</td>
      <td>2.458496e+06</td>
      <td>2.458499e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>17116</th>
      <td>ref_044_14110425-G0013_323735_10140</td>
      <td>2.458496e+06</td>
      <td>2.458499e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>9592</th>
      <td>ref_044_00940255-G0013_1273410_7321</td>
      <td>2.458494e+06</td>
      <td>2.458501e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>15800</th>
      <td>ref_044_14110425-G0013_318062_6704</td>
      <td>2.458496e+06</td>
      <td>2.458499e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>15300</th>
      <td>ref_044_12230255-G0013_304293_4238</td>
      <td>2.458485e+06</td>
      <td>2.458485e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>18548</th>
      <td>ref_044_14110425-G0013_327137_6672</td>
      <td>2.458496e+06</td>
      <td>2.458499e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>16157</th>
      <td>ref_044_14110425-G0013_321195_22023</td>
      <td>2.458496e+06</td>
      <td>2.458499e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>4493</th>
      <td>ref_043_00940255-G0013_1267593_23600</td>
      <td>2.458486e+06</td>
      <td>2.458486e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>14450</th>
      <td>ref_044_12010765-G0013_501847_90690</td>
      <td>2.458478e+06</td>
      <td>2.458478e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>5738</th>
      <td>ref_043_04700255-G0013_27969_13147</td>
      <td>2.458509e+06</td>
      <td>2.458512e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>19733</th>
      <td>ref_044_14110425-G0013_414293_19782</td>
      <td>2.458496e+06</td>
      <td>2.458499e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>8127</th>
      <td>ref_043_12010765-G0013_496916_86866</td>
      <td>2.458519e+06</td>
      <td>2.458526e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>16052</th>
      <td>ref_044_14110425-G0013_320949_13736</td>
      <td>2.458496e+06</td>
      <td>2.458499e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>5672</th>
      <td>ref_043_04700255-G0013_27711_11506</td>
      <td>2.458488e+06</td>
      <td>2.458490e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>19313</th>
      <td>ref_044_14110425-G0013_412983_28297</td>
      <td>2.458496e+06</td>
      <td>2.458499e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>10825</th>
      <td>ref_044_04500085-G0013_1137840_8821</td>
      <td>2.458507e+06</td>
      <td>2.458511e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>570</th>
      <td>ref_022_14110255-G0013_363312_33800</td>
      <td>2.458565e+06</td>
      <td>2.458566e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>18083</th>
      <td>ref_044_14110425-G0013_326188_18287</td>
      <td>2.458496e+06</td>
      <td>2.458499e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>16607</th>
      <td>ref_044_14110425-G0013_322790_22276</td>
      <td>2.458496e+06</td>
      <td>2.458499e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>6080</th>
      <td>ref_043_04700255-G0013_29553_36981</td>
      <td>2.458488e+06</td>
      <td>2.458490e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>7664</th>
      <td>ref_043_12010765-G0013_474975_91886</td>
      <td>2.458519e+06</td>
      <td>2.458526e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>19587</th>
      <td>ref_044_14110425-G0013_414018_15535</td>
      <td>2.458496e+06</td>
      <td>2.458499e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>17043</th>
      <td>ref_044_14110425-G0013_323564_21418</td>
      <td>2.458496e+06</td>
      <td>2.458499e+06</td>
      <td>flare star</td>
    </tr>
  </tbody>
</table>
</div>




```python
#以下是其他一系列尝试，包括使用不同的识别LeNet版本和不同的阈值筛选，
#总体自回归系数小于0.8，模型softmax大于0.98的效果比较好，
#另外还可加入是否拼接的判断，在数量较多情况下进一步筛选
```


```python
final_predicts.is_overlap_bp.value_counts()
```




    N    36
    Y    13
    Name: is_overlap_bp, dtype: int64




```python
final_predicts.loc[final_predicts.is_overlap_bp=='Y','starid'].to_list()
```




    ['ref_044_12870595-G0013_448543_22840',
     'ref_043_06300085-G0013_1457022_9648',
     'ref_043_06300085-G0013_1491366_16831',
     'ref_043_06300085-G0013_1457980_15779',
     'ref_022_13500085-G0014_288288_19986',
     'ref_043_04700255-G0013_29528_21239',
     'ref_043_06300085-G0013_1469618_32402',
     'ref_043_12010765-G0013_475112_87655',
     'ref_043_12010765-G0013_497481_81201',
     'ref_043_04700255-G0013_28874_19479',
     'ref_043_06300085-G0013_1469081_18682',
     'ref_043_12010765-G0013_475075_29192',
     'ref_043_12010765-G0013_474527_203568']




```python
final_predicts.sort_values(by='score',ascending=False,inplace=True)
```


```python
final_predicts.loc[(final_predicts.auto_corr>0.8) & (final_predicts.score>0.5),'starid'].to_list()
```




    ['ref_034_21640255-G0013_569111_13375',
     'ref_022_13500085-G0014_288288_19986',
     'ref_043_12010765-G0013_497481_81201',
     'ref_043_06300085-G0013_1469081_18682',
     'ref_044_02700085-G0013_1154500_44491']




```python
final_predicts.loc[(final_predicts.auto_corr<0.8) & (final_predicts.score>0.5)].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>starid</th>
      <th>event</th>
      <th>score</th>
      <th>path</th>
      <th>l_pos</th>
      <th>r_pos</th>
      <th>l_pos_rel</th>
      <th>r_pos_rel</th>
      <th>auto_corr</th>
      <th>breakpoints</th>
      <th>is_overlap_bp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>166</th>
      <td>ref_021_24960425-G0014_641731_14386</td>
      <td>flare star</td>
      <td>1.000000</td>
      <td>./Finded_0909\ref_021_24960425-G0014_641731_14...</td>
      <td>2.45861e+06</td>
      <td>2.45861e+06</td>
      <td>804</td>
      <td>1004</td>
      <td>0.605088</td>
      <td>[0, 48, 338, 679, 1409, 1480, 1767, 1906, 2445]</td>
      <td>N</td>
    </tr>
    <tr>
      <th>8124</th>
      <td>ref_043_12010765-G0013_496916_197051</td>
      <td>microlensing</td>
      <td>1.000000</td>
      <td>./Finded_0909\ref_043_12010765-G0013_496916_19...</td>
      <td>2.45852e+06</td>
      <td>2.45853e+06</td>
      <td>1147</td>
      <td>1347</td>
      <td>0.418682</td>
      <td>[0, 369, 460, 481, 811, 1015, 1232, 1363]</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>15522</th>
      <td>ref_044_12870595-G0013_448543_22840</td>
      <td>flare star</td>
      <td>1.000000</td>
      <td>./Finded_0909\ref_044_12870595-G0013_448543_22...</td>
      <td>2.45849e+06</td>
      <td>2.4585e+06</td>
      <td>2794</td>
      <td>2994</td>
      <td>0.757787</td>
      <td>[0, 1054, 1435, 1890, 2465, 2893, 3057, 3089, ...</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>570</th>
      <td>ref_022_14110255-G0013_363312_33800</td>
      <td>microlensing</td>
      <td>0.999999</td>
      <td>./Finded_0909\ref_022_14110255-G0013_363312_33...</td>
      <td>2.45857e+06</td>
      <td>2.45857e+06</td>
      <td>929</td>
      <td>1129</td>
      <td>0.570721</td>
      <td>[0, 317, 617, 1113, 1526]</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3194</th>
      <td>ref_041_16810765-G0013_510950_21479</td>
      <td>flare star</td>
      <td>0.999999</td>
      <td>./Finded_0909\ref_041_16810765-G0013_510950_21...</td>
      <td>2.45851e+06</td>
      <td>2.45851e+06</td>
      <td>26</td>
      <td>226</td>
      <td>0.459596</td>
      <td>[0, 620, 627, 1424]</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
</div>




```python
final_predicts.loc[(final_predicts.auto_corr<0.8) & \
                   (final_predicts.score>0.5) & \
                   (final_predicts.is_overlap_bp=='N'),'starid'].to_list()
```




    ['ref_021_24960425-G0014_641731_14386',
     'ref_043_12010765-G0013_496916_197051',
     'ref_022_14110255-G0013_363312_33800',
     'ref_041_16810765-G0013_510950_21479',
     'ref_043_12010765-G0013_474914_29466',
     'ref_044_34820255-G0013_1288349_20418',
     'ref_043_12010765-G0013_474975_91886',
     'ref_043_00940255-G0013_131769_27879',
     'ref_043_12010765-G0013_474944_91913',
     'ref_044_12010765-G0013_131065_100362',
     'ref_044_18590595-G0013_382080_13703',
     'ref_044_12010765-G0013_500055_12380',
     'ref_044_04700255-G0013_49380_2518',
     'ref_043_12010765-G0013_474910_90832',
     'ref_044_00940255-G0013_1273778_7291',
     'ref_043_04700255-G0013_28382_6790',
     'ref_043_00940255-G0013_1178192_26131',
     'ref_044_34820255-G0013_1288250_20618',
     'ref_043_04700255-G0013_25654_19167',
     'ref_044_00940255-G0013_1273574_8915',
     'ref_043_04700255-G0013_28122_34945',
     'ref_044_34820255-G0013_808001_35420',
     'ref_043_16810765-G0013_762424_95233',
     'ref_044_34820255-G0013_1287904_7965',
     'ref_044_34820255-G0013_852269_24303',
     'ref_043_12010765-G0013_474889_90556',
     'ref_041_21450595-G0013_717252_21975',
     'ref_044_00940255-G0013_1273010_7276',
     'ref_043_12010765-G0013_509175_728053',
     'ref_044_00940255-G0013_1283463_27669',
     'ref_043_04700255-G0013_25669_16634',
     'ref_022_14110425-G0013_366470_14934',
     'ref_044_14110425-G0013_322955_20317',
     'ref_044_04700255-G0013_38628_11061']




```python
final_predicts.loc[(final_predicts.auto_corr<0.8) & \
                   (final_predicts.score>0.5) & \
                   (final_predicts.is_overlap_bp=='N'),\
                   ['starid','l_pos', 'r_pos','event','score']].score.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x266d33abb88>




![png](output_28_1.png)



```python
final_predicts.loc[(final_predicts.auto_corr<0.8) & \
                   (final_predicts.score>0.5) & \
                   (final_predicts.is_overlap_bp=='N'),\
                   ['starid','l_pos', 'r_pos','event','score']].event.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x266d4f04208>




![png](output_29_1.png)



```python
len(final_predicts.loc[(final_predicts.auto_corr<0.8) & \
                   (final_predicts.score>0.5) & \
                   (final_predicts.is_overlap_bp=='N'),\
                   ['starid','l_pos', 'r_pos','event','score']])
```




    34




```python
final_predicts
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>starid</th>
      <th>event</th>
      <th>score</th>
      <th>path</th>
      <th>l_pos</th>
      <th>r_pos</th>
      <th>l_pos_rel</th>
      <th>r_pos_rel</th>
      <th>auto_corr</th>
      <th>breakpoints</th>
      <th>is_overlap_bp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>166</th>
      <td>ref_021_24960425-G0014_641731_14386</td>
      <td>flare star</td>
      <td>1.000000</td>
      <td>./Finded_0909\ref_021_24960425-G0014_641731_14...</td>
      <td>2.45861e+06</td>
      <td>2.45861e+06</td>
      <td>804</td>
      <td>1004</td>
      <td>0.605088</td>
      <td>[0, 48, 338, 679, 1409, 1480, 1767, 1906, 2445]</td>
      <td>N</td>
    </tr>
    <tr>
      <th>8124</th>
      <td>ref_043_12010765-G0013_496916_197051</td>
      <td>microlensing</td>
      <td>1.000000</td>
      <td>./Finded_0909\ref_043_12010765-G0013_496916_19...</td>
      <td>2.45852e+06</td>
      <td>2.45853e+06</td>
      <td>1147</td>
      <td>1347</td>
      <td>0.418682</td>
      <td>[0, 369, 460, 481, 811, 1015, 1232, 1363]</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>15522</th>
      <td>ref_044_12870595-G0013_448543_22840</td>
      <td>flare star</td>
      <td>1.000000</td>
      <td>./Finded_0909\ref_044_12870595-G0013_448543_22...</td>
      <td>2.45849e+06</td>
      <td>2.4585e+06</td>
      <td>2794</td>
      <td>2994</td>
      <td>0.757787</td>
      <td>[0, 1054, 1435, 1890, 2465, 2893, 3057, 3089, ...</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>570</th>
      <td>ref_022_14110255-G0013_363312_33800</td>
      <td>microlensing</td>
      <td>0.999999</td>
      <td>./Finded_0909\ref_022_14110255-G0013_363312_33...</td>
      <td>2.45857e+06</td>
      <td>2.45857e+06</td>
      <td>929</td>
      <td>1129</td>
      <td>0.570721</td>
      <td>[0, 317, 617, 1113, 1526]</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3194</th>
      <td>ref_041_16810765-G0013_510950_21479</td>
      <td>flare star</td>
      <td>0.999999</td>
      <td>./Finded_0909\ref_041_16810765-G0013_510950_21...</td>
      <td>2.45851e+06</td>
      <td>2.45851e+06</td>
      <td>26</td>
      <td>226</td>
      <td>0.459596</td>
      <td>[0, 620, 627, 1424]</td>
      <td>N</td>
    </tr>
    <tr>
      <th>1156</th>
      <td>ref_034_21640255-G0013_569111_13375</td>
      <td>microlensing</td>
      <td>0.999958</td>
      <td>./Finded_0909\ref_034_21640255-G0013_569111_13...</td>
      <td>2.45852e+06</td>
      <td>2.45852e+06</td>
      <td>70</td>
      <td>270</td>
      <td>0.857123</td>
      <td>[0, 115, 429, 1079]</td>
      <td>N</td>
    </tr>
    <tr>
      <th>7604</th>
      <td>ref_043_12010765-G0013_474914_29466</td>
      <td>flare star</td>
      <td>0.999944</td>
      <td>./Finded_0909\ref_043_12010765-G0013_474914_29...</td>
      <td>2.45852e+06</td>
      <td>2.45853e+06</td>
      <td>2686</td>
      <td>2886</td>
      <td>0.406504</td>
      <td>[0, 981, 1308, 1653, 2123, 2473, 2761, 2980]</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6212</th>
      <td>ref_043_06300085-G0013_1457022_9648</td>
      <td>flare star</td>
      <td>0.999924</td>
      <td>./Finded_0909\ref_043_06300085-G0013_1457022_9...</td>
      <td>2.45852e+06</td>
      <td>2.45852e+06</td>
      <td>2570</td>
      <td>2770</td>
      <td>0.627099</td>
      <td>[0, 948, 958, 1271, 1921, 2669, 3087]</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>21151</th>
      <td>ref_044_34820255-G0013_1288349_20418</td>
      <td>flare star</td>
      <td>0.999861</td>
      <td>./Finded_0909\ref_044_34820255-G0013_1288349_2...</td>
      <td>2.4585e+06</td>
      <td>2.4585e+06</td>
      <td>506</td>
      <td>706</td>
      <td>0.601350</td>
      <td>[0, 486, 888, 1233]</td>
      <td>N</td>
    </tr>
    <tr>
      <th>6680</th>
      <td>ref_043_06300085-G0013_1491366_16831</td>
      <td>flare star</td>
      <td>0.999759</td>
      <td>./Finded_0909\ref_043_06300085-G0013_1491366_1...</td>
      <td>2.45852e+06</td>
      <td>2.45852e+06</td>
      <td>2571</td>
      <td>2771</td>
      <td>0.685842</td>
      <td>[0, 948, 958, 1271, 1921, 2669, 3087]</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>7664</th>
      <td>ref_043_12010765-G0013_474975_91886</td>
      <td>flare star</td>
      <td>0.999413</td>
      <td>./Finded_0909\ref_043_12010765-G0013_474975_91...</td>
      <td>2.45852e+06</td>
      <td>2.45853e+06</td>
      <td>1246</td>
      <td>1446</td>
      <td>0.474819</td>
      <td>[0, 335, 638, 894, 1076, 1115, 1331, 1515]</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>4563</th>
      <td>ref_043_00940255-G0013_131769_27879</td>
      <td>flare star</td>
      <td>0.999375</td>
      <td>./Finded_0909\ref_043_00940255-G0013_131769_27...</td>
      <td>2.45849e+06</td>
      <td>2.45849e+06</td>
      <td>741</td>
      <td>941</td>
      <td>0.667187</td>
      <td>[0, 589, 1198, 1359]</td>
      <td>N</td>
    </tr>
    <tr>
      <th>6298</th>
      <td>ref_043_06300085-G0013_1457980_15779</td>
      <td>flare star</td>
      <td>0.998850</td>
      <td>./Finded_0909\ref_043_06300085-G0013_1457980_1...</td>
      <td>2.45852e+06</td>
      <td>2.45852e+06</td>
      <td>2570</td>
      <td>2770</td>
      <td>0.697776</td>
      <td>[0, 948, 958, 1271, 1921, 2669, 3087]</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>7632</th>
      <td>ref_043_12010765-G0013_474944_91913</td>
      <td>flare star</td>
      <td>0.997329</td>
      <td>./Finded_0909\ref_043_12010765-G0013_474944_91...</td>
      <td>2.45852e+06</td>
      <td>2.45853e+06</td>
      <td>2600</td>
      <td>2800</td>
      <td>0.337942</td>
      <td>[0, 869, 1195, 1523, 1932, 2248, 2694, 3027]</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>507</th>
      <td>ref_022_13500085-G0014_288288_19986</td>
      <td>flare star</td>
      <td>0.996597</td>
      <td>./Finded_0909\ref_022_13500085-G0014_288288_19...</td>
      <td>2.45855e+06</td>
      <td>2.45855e+06</td>
      <td>398</td>
      <td>598</td>
      <td>0.843329</td>
      <td>[0, 497, 1211, 1705, 1771]</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6063</th>
      <td>ref_043_04700255-G0013_29528_21239</td>
      <td>flare star</td>
      <td>0.995580</td>
      <td>./Finded_0909\ref_043_04700255-G0013_29528_212...</td>
      <td>2.45851e+06</td>
      <td>2.45852e+06</td>
      <td>1811</td>
      <td>2011</td>
      <td>0.508459</td>
      <td>[0, 606, 755, 1219, 1909, 2215, 2426, 2774]</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6485</th>
      <td>ref_043_06300085-G0013_1469618_32402</td>
      <td>flare star</td>
      <td>0.994973</td>
      <td>./Finded_0909\ref_043_06300085-G0013_1469618_3...</td>
      <td>2.45851e+06</td>
      <td>2.45852e+06</td>
      <td>1825</td>
      <td>2025</td>
      <td>0.419319</td>
      <td>[0, 948, 958, 1271, 1920, 1924, 2141]</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>13338</th>
      <td>ref_044_12010765-G0013_131065_100362</td>
      <td>microlensing</td>
      <td>0.994560</td>
      <td>./Finded_0909\ref_044_12010765-G0013_131065_10...</td>
      <td>2.45848e+06</td>
      <td>2.45849e+06</td>
      <td>602</td>
      <td>802</td>
      <td>0.492758</td>
      <td>[0, 191, 750, 1884]</td>
      <td>N</td>
    </tr>
    <tr>
      <th>20298</th>
      <td>ref_044_18590595-G0013_382080_13703</td>
      <td>flare star</td>
      <td>0.994028</td>
      <td>./Finded_0909\ref_044_18590595-G0013_382080_13...</td>
      <td>2.45849e+06</td>
      <td>2.45849e+06</td>
      <td>213</td>
      <td>413</td>
      <td>0.331218</td>
      <td>[0, 304, 601, 1094]</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>7752</th>
      <td>ref_043_12010765-G0013_475112_87655</td>
      <td>flare star</td>
      <td>0.993346</td>
      <td>./Finded_0909\ref_043_12010765-G0013_475112_87...</td>
      <td>2.45852e+06</td>
      <td>2.45853e+06</td>
      <td>2984</td>
      <td>3184</td>
      <td>0.368048</td>
      <td>[0, 935, 1226, 1552, 2316, 2660, 3081, 3373]</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>14009</th>
      <td>ref_044_12010765-G0013_500055_12380</td>
      <td>microlensing</td>
      <td>0.985325</td>
      <td>./Finded_0909\ref_044_12010765-G0013_500055_12...</td>
      <td>2.45847e+06</td>
      <td>2.45847e+06</td>
      <td>74</td>
      <td>274</td>
      <td>0.320638</td>
      <td>[0, 66, 636, 1277, 2503, 2878, 3657, 4039, 469...</td>
      <td>N</td>
    </tr>
    <tr>
      <th>13108</th>
      <td>ref_044_04700255-G0013_49380_2518</td>
      <td>flare star</td>
      <td>0.979079</td>
      <td>./Finded_0909\ref_044_04700255-G0013_49380_251...</td>
      <td>2.45849e+06</td>
      <td>2.45849e+06</td>
      <td>2251</td>
      <td>2451</td>
      <td>0.336639</td>
      <td>[0, 2, 316, 1185, 1803, 1814, 2384, 2927, 2936...</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>7597</th>
      <td>ref_043_12010765-G0013_474910_90832</td>
      <td>flare star</td>
      <td>0.976021</td>
      <td>./Finded_0909\ref_043_12010765-G0013_474910_90...</td>
      <td>2.45852e+06</td>
      <td>2.45853e+06</td>
      <td>1817</td>
      <td>2017</td>
      <td>0.371039</td>
      <td>[0, 601, 796, 983, 1354, 1643, 1903, 2111]</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>9612</th>
      <td>ref_044_00940255-G0013_1273778_7291</td>
      <td>microlensing</td>
      <td>0.973477</td>
      <td>./Finded_0909\ref_044_00940255-G0013_1273778_7...</td>
      <td>2.4585e+06</td>
      <td>2.4585e+06</td>
      <td>3017</td>
      <td>3217</td>
      <td>0.726226</td>
      <td>[0, 595, 1464, 1585, 2194, 2873, 3157, 3443, 3...</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>8220</th>
      <td>ref_043_12010765-G0013_497481_81201</td>
      <td>flare star</td>
      <td>0.971844</td>
      <td>./Finded_0909\ref_043_12010765-G0013_497481_81...</td>
      <td>2.4585e+06</td>
      <td>2.45851e+06</td>
      <td>518</td>
      <td>718</td>
      <td>0.947814</td>
      <td>[0, 522, 617, 719, 1033, 1362, 1609, 1785]</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>5839</th>
      <td>ref_043_04700255-G0013_28382_6790</td>
      <td>microlensing</td>
      <td>0.969493</td>
      <td>./Finded_0909\ref_043_04700255-G0013_28382_679...</td>
      <td>2.45851e+06</td>
      <td>2.45851e+06</td>
      <td>831</td>
      <td>1031</td>
      <td>0.442348</td>
      <td>[0, 711, 860, 1323, 1626, 1886, 1888]</td>
      <td>N</td>
    </tr>
    <tr>
      <th>5902</th>
      <td>ref_043_04700255-G0013_28874_19479</td>
      <td>flare star</td>
      <td>0.966021</td>
      <td>./Finded_0909\ref_043_04700255-G0013_28874_194...</td>
      <td>2.45851e+06</td>
      <td>2.45851e+06</td>
      <td>1225</td>
      <td>1425</td>
      <td>0.490155</td>
      <td>[0, 711, 860, 1324, 2014, 2320, 2580, 2928]</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>4452</th>
      <td>ref_043_00940255-G0013_1178192_26131</td>
      <td>flare star</td>
      <td>0.961358</td>
      <td>./Finded_0909\ref_043_00940255-G0013_1178192_2...</td>
      <td>2.45849e+06</td>
      <td>2.45849e+06</td>
      <td>731</td>
      <td>931</td>
      <td>0.740834</td>
      <td>[0, 589, 1198, 1558]</td>
      <td>N</td>
    </tr>
    <tr>
      <th>21103</th>
      <td>ref_044_34820255-G0013_1288250_20618</td>
      <td>flare star</td>
      <td>0.955150</td>
      <td>./Finded_0909\ref_044_34820255-G0013_1288250_2...</td>
      <td>2.4585e+06</td>
      <td>2.4585e+06</td>
      <td>508</td>
      <td>708</td>
      <td>0.481095</td>
      <td>[0, 486, 889, 1234]</td>
      <td>N</td>
    </tr>
    <tr>
      <th>5330</th>
      <td>ref_043_04700255-G0013_25654_19167</td>
      <td>flare star</td>
      <td>0.943077</td>
      <td>./Finded_0909\ref_043_04700255-G0013_25654_191...</td>
      <td>2.45849e+06</td>
      <td>2.45849e+06</td>
      <td>1547</td>
      <td>1747</td>
      <td>0.313230</td>
      <td>[0, 0, 1, 314, 1101, 1697, 2243, 2251]</td>
      <td>N</td>
    </tr>
    <tr>
      <th>9603</th>
      <td>ref_044_00940255-G0013_1273574_8915</td>
      <td>microlensing</td>
      <td>0.918755</td>
      <td>./Finded_0909\ref_044_00940255-G0013_1273574_8...</td>
      <td>2.4585e+06</td>
      <td>2.4585e+06</td>
      <td>3021</td>
      <td>3221</td>
      <td>0.364824</td>
      <td>[0, 595, 1464, 1585, 2194, 2873, 3157, 3443, 3...</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>5786</th>
      <td>ref_043_04700255-G0013_28122_34945</td>
      <td>microlensing</td>
      <td>0.901100</td>
      <td>./Finded_0909\ref_043_04700255-G0013_28122_349...</td>
      <td>2.45849e+06</td>
      <td>2.45849e+06</td>
      <td>1847</td>
      <td>2047</td>
      <td>0.433016</td>
      <td>[0, 66, 418, 1368, 2002, 2644, 2652]</td>
      <td>N</td>
    </tr>
    <tr>
      <th>6443</th>
      <td>ref_043_06300085-G0013_1469081_18682</td>
      <td>flare star</td>
      <td>0.884702</td>
      <td>./Finded_0909\ref_043_06300085-G0013_1469081_1...</td>
      <td>2.45852e+06</td>
      <td>2.45852e+06</td>
      <td>2570</td>
      <td>2770</td>
      <td>0.837384</td>
      <td>[0, 948, 958, 1271, 1921, 2669, 3087]</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>21433</th>
      <td>ref_044_34820255-G0013_808001_35420</td>
      <td>microlensing</td>
      <td>0.851976</td>
      <td>./Finded_0909\ref_044_34820255-G0013_808001_35...</td>
      <td>2.4585e+06</td>
      <td>2.4585e+06</td>
      <td>520</td>
      <td>720</td>
      <td>0.577183</td>
      <td>[0, 483, 886, 1232]</td>
      <td>N</td>
    </tr>
    <tr>
      <th>8964</th>
      <td>ref_043_16810765-G0013_762424_95233</td>
      <td>microlensing</td>
      <td>0.842528</td>
      <td>./Finded_0909\ref_043_16810765-G0013_762424_95...</td>
      <td>2.45851e+06</td>
      <td>2.45851e+06</td>
      <td>840</td>
      <td>1040</td>
      <td>0.650792</td>
      <td>[0, 682, 690, 1472]</td>
      <td>N</td>
    </tr>
    <tr>
      <th>21065</th>
      <td>ref_044_34820255-G0013_1287904_7965</td>
      <td>microlensing</td>
      <td>0.795483</td>
      <td>./Finded_0909\ref_044_34820255-G0013_1287904_7...</td>
      <td>2.4585e+06</td>
      <td>2.4585e+06</td>
      <td>515</td>
      <td>715</td>
      <td>0.468203</td>
      <td>[0, 486, 889, 1235]</td>
      <td>N</td>
    </tr>
    <tr>
      <th>21545</th>
      <td>ref_044_34820255-G0013_852269_24303</td>
      <td>flare star</td>
      <td>0.782227</td>
      <td>./Finded_0909\ref_044_34820255-G0013_852269_24...</td>
      <td>2.4585e+06</td>
      <td>2.4585e+06</td>
      <td>530</td>
      <td>730</td>
      <td>0.642200</td>
      <td>[0, 486, 889, 1235]</td>
      <td>N</td>
    </tr>
    <tr>
      <th>7584</th>
      <td>ref_043_12010765-G0013_474889_90556</td>
      <td>microlensing</td>
      <td>0.776479</td>
      <td>./Finded_0909\ref_043_12010765-G0013_474889_90...</td>
      <td>2.45852e+06</td>
      <td>2.45853e+06</td>
      <td>2893</td>
      <td>3093</td>
      <td>0.511600</td>
      <td>[0, 977, 1349, 1637, 2384, 2523, 2989, 3544]</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>3892</th>
      <td>ref_041_21450595-G0013_717252_21975</td>
      <td>flare star</td>
      <td>0.713862</td>
      <td>./Finded_0909\ref_041_21450595-G0013_717252_21...</td>
      <td>2.45851e+06</td>
      <td>2.45851e+06</td>
      <td>439</td>
      <td>639</td>
      <td>0.602697</td>
      <td>[0, 233, 601, 828, 1561, 1603]</td>
      <td>N</td>
    </tr>
    <tr>
      <th>9574</th>
      <td>ref_044_00940255-G0013_1273010_7276</td>
      <td>microlensing</td>
      <td>0.688334</td>
      <td>./Finded_0909\ref_044_00940255-G0013_1273010_7...</td>
      <td>2.45849e+06</td>
      <td>2.4585e+06</td>
      <td>2781</td>
      <td>2981</td>
      <td>0.212791</td>
      <td>[0, 595, 1464, 1585, 2194, 2873, 3096, 3382, 3...</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>7736</th>
      <td>ref_043_12010765-G0013_475075_29192</td>
      <td>flare star</td>
      <td>0.654178</td>
      <td>./Finded_0909\ref_043_12010765-G0013_475075_29...</td>
      <td>2.45852e+06</td>
      <td>2.45853e+06</td>
      <td>1512</td>
      <td>1712</td>
      <td>0.532780</td>
      <td>[0, 450, 676, 842, 1372, 1395, 1609, 2007]</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>9869</th>
      <td>ref_044_02700085-G0013_1154500_44491</td>
      <td>microlensing</td>
      <td>0.653499</td>
      <td>./Finded_0909\ref_044_02700085-G0013_1154500_4...</td>
      <td>2.45848e+06</td>
      <td>2.45848e+06</td>
      <td>318</td>
      <td>518</td>
      <td>0.991616</td>
      <td>[0, 10, 252, 269, 1008]</td>
      <td>N</td>
    </tr>
    <tr>
      <th>8637</th>
      <td>ref_043_12010765-G0013_509175_728053</td>
      <td>microlensing</td>
      <td>0.639736</td>
      <td>./Finded_0909\ref_043_12010765-G0013_509175_72...</td>
      <td>2.45849e+06</td>
      <td>2.45849e+06</td>
      <td>424</td>
      <td>624</td>
      <td>0.714637</td>
      <td>[0, 1090]</td>
      <td>N</td>
    </tr>
    <tr>
      <th>9634</th>
      <td>ref_044_00940255-G0013_1283463_27669</td>
      <td>microlensing</td>
      <td>0.628939</td>
      <td>./Finded_0909\ref_044_00940255-G0013_1283463_2...</td>
      <td>2.45849e+06</td>
      <td>2.4585e+06</td>
      <td>587</td>
      <td>787</td>
      <td>0.354716</td>
      <td>[0, 678, 917, 1203, 1376]</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>5336</th>
      <td>ref_043_04700255-G0013_25669_16634</td>
      <td>flare star</td>
      <td>0.619645</td>
      <td>./Finded_0909\ref_043_04700255-G0013_25669_166...</td>
      <td>2.45852e+06</td>
      <td>2.45852e+06</td>
      <td>2058</td>
      <td>2258</td>
      <td>0.189898</td>
      <td>[0, 711, 860, 1324, 2014, 2320, 2577, 2924]</td>
      <td>N</td>
    </tr>
    <tr>
      <th>578</th>
      <td>ref_022_14110425-G0013_366470_14934</td>
      <td>microlensing</td>
      <td>0.586759</td>
      <td>./Finded_0909\ref_022_14110425-G0013_366470_14...</td>
      <td>2.45851e+06</td>
      <td>2.45854e+06</td>
      <td>2098</td>
      <td>2298</td>
      <td>0.342597</td>
      <td>[0, 471, 1675, 2215, 2222, 2610]</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>7452</th>
      <td>ref_043_12010765-G0013_474527_203568</td>
      <td>flare star</td>
      <td>0.561159</td>
      <td>./Finded_0909\ref_043_12010765-G0013_474527_20...</td>
      <td>2.45852e+06</td>
      <td>2.45853e+06</td>
      <td>2664</td>
      <td>2864</td>
      <td>0.407665</td>
      <td>[0, 369, 460, 1176, 1520, 1863, 2161, 2383, 27...</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>16705</th>
      <td>ref_044_14110425-G0013_322955_20317</td>
      <td>microlensing</td>
      <td>0.540820</td>
      <td>./Finded_0909\ref_044_14110425-G0013_322955_20...</td>
      <td>2.45853e+06</td>
      <td>2.45853e+06</td>
      <td>6463</td>
      <td>6663</td>
      <td>0.750876</td>
      <td>[0, 284, 379, 1483, 1731, 2223, 2918, 3673, 44...</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>12571</th>
      <td>ref_044_04700255-G0013_38628_11061</td>
      <td>flare star</td>
      <td>0.527065</td>
      <td>./Finded_0909\ref_044_04700255-G0013_38628_110...</td>
      <td>2.45849e+06</td>
      <td>2.45849e+06</td>
      <td>801</td>
      <td>1001</td>
      <td>0.334629</td>
      <td>[0, 8, 78, 458, 549, 557, 934, 1037, 1043, 131...</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>
</div>




```python
final_predicts.loc[(final_predicts.auto_corr<0.8) & \
                   (final_predicts.score>0.5) & \
                   (final_predicts.is_overlap_bp=='N'),\
                   ['starid','l_pos', 'r_pos','event']].to_csv('./res_20200912.csv',header=False,index=False)
```


```python
final_predicts[['starid','l_pos', 'r_pos','event']].to_csv('./res_20200909.csv',header=False,index=False)
```


```python
final_predicts.loc[(final_predicts.auto_corr<0.8) & (final_predicts.score>0.98),\
                   ['starid','l_pos', 'r_pos','event']].head(50).to_csv('./res_20200909.csv',header=False,index=False)
```


```python
final_predicts.loc[(final_predicts.auto_corr<0.8) & (final_predicts.score>0.5),\
                   ['starid','l_pos', 'r_pos','event']].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>starid</th>
      <th>l_pos</th>
      <th>r_pos</th>
      <th>event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>166</th>
      <td>ref_021_24960425-G0014_641731_14386</td>
      <td>2.45861e+06</td>
      <td>2.45861e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>8124</th>
      <td>ref_043_12010765-G0013_496916_197051</td>
      <td>2.45852e+06</td>
      <td>2.45853e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>15522</th>
      <td>ref_044_12870595-G0013_448543_22840</td>
      <td>2.45849e+06</td>
      <td>2.4585e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>570</th>
      <td>ref_022_14110255-G0013_363312_33800</td>
      <td>2.45857e+06</td>
      <td>2.45857e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>3194</th>
      <td>ref_041_16810765-G0013_510950_21479</td>
      <td>2.45851e+06</td>
      <td>2.45851e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>7604</th>
      <td>ref_043_12010765-G0013_474914_29466</td>
      <td>2.45852e+06</td>
      <td>2.45853e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>6212</th>
      <td>ref_043_06300085-G0013_1457022_9648</td>
      <td>2.45852e+06</td>
      <td>2.45852e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>21151</th>
      <td>ref_044_34820255-G0013_1288349_20418</td>
      <td>2.4585e+06</td>
      <td>2.4585e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>6680</th>
      <td>ref_043_06300085-G0013_1491366_16831</td>
      <td>2.45852e+06</td>
      <td>2.45852e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>7664</th>
      <td>ref_043_12010765-G0013_474975_91886</td>
      <td>2.45852e+06</td>
      <td>2.45853e+06</td>
      <td>flare star</td>
    </tr>
  </tbody>
</table>
</div>




```python
final_predicts.loc[(final_predicts.auto_corr<0.8) & (final_predicts.score>0.98),\
                   ['starid','l_pos', 'r_pos','event']].starid.to_list()
```




    ['ref_021_24960425-G0014_641731_14386',
     'ref_044_12870595-G0013_448543_22840',
     'ref_043_12010765-G0013_496916_197051',
     'ref_044_00940255-G0013_1273778_7291',
     'ref_043_04700255-G0013_29528_21239',
     'ref_043_00940255-G0013_1265997_24761',
     'ref_043_04700255-G0013_28874_19479',
     'ref_043_12010765-G0013_474914_29466',
     'ref_041_16810765-G0013_510950_21479',
     'ref_044_14110425-G0013_410983_22399',
     'ref_043_04700255-G0013_28382_6790',
     'ref_044_18590595-G0013_382080_13703',
     'ref_044_04700255-G0013_45239_27087',
     'ref_043_04700255-G0013_28122_34945',
     'ref_043_06300085-G0013_1457022_9648',
     'ref_044_14110425-G0013_326441_15380',
     'ref_044_12010765-G0013_500055_12380',
     'ref_044_34820255-G0013_1287904_7965',
     'ref_043_06300085-G0013_1491366_16831',
     'ref_044_04500085-G0013_1133981_14981',
     'ref_043_06300085-G0013_1469618_32402',
     'ref_044_12230255-G0013_1441255_28735',
     'ref_043_12010765-G0013_474910_90832',
     'ref_044_14110425-G0013_324602_9737',
     'ref_044_14110425-G0013_321481_15782',
     'ref_044_34820255-G0013_1288349_20418',
     'ref_044_14110425-G0013_321183_19981',
     'ref_044_14110425-G0013_325266_7324',
     'ref_044_14110425-G0013_323735_10140',
     'ref_044_00940255-G0013_1273410_7321',
     'ref_044_14110425-G0013_318062_6704',
     'ref_044_12230255-G0013_304293_4238',
     'ref_044_14110425-G0013_327137_6672',
     'ref_044_14110425-G0013_321195_22023',
     'ref_043_00940255-G0013_1267593_23600',
     'ref_044_12010765-G0013_501847_90690',
     'ref_043_04700255-G0013_27969_13147',
     'ref_044_14110425-G0013_414293_19782',
     'ref_043_12010765-G0013_496916_86866',
     'ref_044_14110425-G0013_320949_13736',
     'ref_043_04700255-G0013_27711_11506',
     'ref_044_14110425-G0013_412983_28297',
     'ref_044_04500085-G0013_1137840_8821',
     'ref_022_14110255-G0013_363312_33800',
     'ref_044_14110425-G0013_326188_18287',
     'ref_044_14110425-G0013_322790_22276',
     'ref_043_04700255-G0013_29553_36981',
     'ref_043_12010765-G0013_474975_91886',
     'ref_044_14110425-G0013_414018_15535',
     'ref_044_14110425-G0013_323564_21418',
     'ref_044_00940255-G0013_1259286_27637',
     'ref_044_14110425-G0013_410680_28021',
     'ref_044_14110425-G0013_413076_28961',
     'ref_044_14110425-G0013_411438_27339',
     'ref_043_12010765-G0013_475112_87655',
     'ref_044_14110425-G0013_320831_13055']




```python
final_predicts[['starid','l_pos', 'r_pos','event']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>starid</th>
      <th>l_pos</th>
      <th>r_pos</th>
      <th>event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1194</th>
      <td>ref_022_15730595-G0013_481532_2590</td>
      <td>2.45851e+06</td>
      <td>2.45851e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>1776</th>
      <td>ref_024_13500085-G0013_1397642_27534</td>
      <td>2.4585e+06</td>
      <td>2.4585e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>637</th>
      <td>ref_021_24310595-G0013_746468_14893</td>
      <td>2.45858e+06</td>
      <td>2.45858e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>7418</th>
      <td>ref_031_12010765-G0013_517053_76902</td>
      <td>2.45851e+06</td>
      <td>2.45853e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>62</th>
      <td>ref_021_18450425-G0014_352007_14642</td>
      <td>2.45858e+06</td>
      <td>2.45858e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>6090</th>
      <td>ref_031_12010765-G0013_251733_237296</td>
      <td>2.45852e+06</td>
      <td>2.45853e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>1156</th>
      <td>ref_022_15730595-G0013_391462_6330</td>
      <td>2.45854e+06</td>
      <td>2.45854e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>9184</th>
      <td>ref_031_12230255-G0013_396537_492471</td>
      <td>2.45848e+06</td>
      <td>2.45849e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>11005</th>
      <td>ref_031_12230255-G0013_401749_506511</td>
      <td>2.45848e+06</td>
      <td>2.45849e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>4090</th>
      <td>ref_031_06300085-G0013_20382_64595</td>
      <td>2.45849e+06</td>
      <td>2.4585e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>192</th>
      <td>ref_021_18450425-G0014_703997_4523</td>
      <td>2.45858e+06</td>
      <td>2.45858e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>8098</th>
      <td>ref_031_12230255-G0013_1616427_1638087</td>
      <td>2.45848e+06</td>
      <td>2.45849e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>10206</th>
      <td>ref_031_12230255-G0013_399656_474332</td>
      <td>2.45848e+06</td>
      <td>2.45848e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>12789</th>
      <td>ref_031_12230255-G0013_406124_493867</td>
      <td>2.45848e+06</td>
      <td>2.45849e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>7424</th>
      <td>ref_031_12010765-G0013_517095_4910</td>
      <td>2.45852e+06</td>
      <td>2.45853e+06</td>
      <td>flare star</td>
    </tr>
    <tr>
      <th>9449</th>
      <td>ref_031_12230255-G0013_396917_492899</td>
      <td>2.45848e+06</td>
      <td>2.45849e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>10455</th>
      <td>ref_031_12230255-G0013_400132_476547</td>
      <td>2.45848e+06</td>
      <td>2.45849e+06</td>
      <td>microlensing</td>
    </tr>
    <tr>
      <th>1834</th>
      <td>ref_031_06300085-G0013_1490780_10263</td>
      <td>2.45849e+06</td>
      <td>2.45849e+06</td>
      <td>microlensing</td>
    </tr>
  </tbody>
</table>
</div>




```python
####End of coding
```


```python
### 以下处理同一个星体被检出两种事件的情况，实际上发生率很低，只是备用
```


```python
predicts_event_lst=predicts.groupby('starid')['event'].apply(lambda x:x.to_list())
predicts_prob_lst=predicts.groupby('starid')['prob'].apply(lambda x:x.to_list())
```


```python
final_pred=pd.concat([predicts_event_lst,predicts_prob_lst],axis=1)
```


```python
final_pred['real_event']=final_pred[['event','prob']].apply(lambda x:x[0][np.argmax(x[1])],axis=1)
```


```python
final_pred.tail(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>event</th>
      <th>prob</th>
      <th>real_event</th>
    </tr>
    <tr>
      <th>starid</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ref_031_12230255-G0013_407699_368581</th>
      <td>[2]</td>
      <td>[1.0]</td>
      <td>2</td>
    </tr>
    <tr>
      <th>ref_031_12230255-G0013_407715_167208</th>
      <td>[2]</td>
      <td>[0.999998927116394]</td>
      <td>2</td>
    </tr>
    <tr>
      <th>ref_031_12230255-G0013_407721_506905</th>
      <td>[2]</td>
      <td>[0.9999998807907104]</td>
      <td>2</td>
    </tr>
    <tr>
      <th>ref_031_12230255-G0013_407736_371627</th>
      <td>[2]</td>
      <td>[0.9987753033638]</td>
      <td>2</td>
    </tr>
    <tr>
      <th>ref_031_12230255-G0013_407751_1178879</th>
      <td>[1]</td>
      <td>[0.6667221188545227]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>ref_031_12230255-G0013_407815_357352</th>
      <td>[1]</td>
      <td>[0.9991493225097656]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>ref_031_12230255-G0013_407818_1173546</th>
      <td>[2]</td>
      <td>[0.827765941619873]</td>
      <td>2</td>
    </tr>
    <tr>
      <th>ref_031_12230255-G0013_407841_358927</th>
      <td>[2]</td>
      <td>[0.9876903295516968]</td>
      <td>2</td>
    </tr>
    <tr>
      <th>ref_031_12230255-G0013_407849_360262</th>
      <td>[2]</td>
      <td>[0.9997221827507019]</td>
      <td>2</td>
    </tr>
    <tr>
      <th>ref_031_12230255-G0013_407873_353228</th>
      <td>[2]</td>
      <td>[0.9999986886978149]</td>
      <td>2</td>
    </tr>
    <tr>
      <th>ref_031_12230255-G0013_408076_374363</th>
      <td>[2]</td>
      <td>[0.5597881078720093]</td>
      <td>2</td>
    </tr>
    <tr>
      <th>ref_031_12230255-G0013_412210_319756</th>
      <td>[1]</td>
      <td>[0.9497370719909668]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>ref_031_12230255-G0013_412211_319540</th>
      <td>[2]</td>
      <td>[0.9999946355819702]</td>
      <td>2</td>
    </tr>
    <tr>
      <th>ref_031_12230255-G0013_412220_1132685</th>
      <td>[1]</td>
      <td>[0.8891890048980713]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>ref_031_14110425-G0013_417555_145653</th>
      <td>[1]</td>
      <td>[0.6431736946105957]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>ref_031_14110425-G0013_423583_161337</th>
      <td>[1]</td>
      <td>[0.9998540878295898]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>ref_031_14110425-G0013_460938_140954</th>
      <td>[2]</td>
      <td>[0.999906063079834]</td>
      <td>2</td>
    </tr>
    <tr>
      <th>ref_031_14110425-G0013_461362_141910</th>
      <td>[2]</td>
      <td>[1.0]</td>
      <td>2</td>
    </tr>
    <tr>
      <th>ref_031_14110425-G0013_467014_143303</th>
      <td>[2]</td>
      <td>[0.999998927116394]</td>
      <td>2</td>
    </tr>
    <tr>
      <th>ref_031_14110425-G0013_467231_139997</th>
      <td>[2]</td>
      <td>[0.9999998807907104]</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
final_pred['real_event']=final_pred['real_event'].replace({2:'microlensing'})
```


```python
final_pred['real_event'].to_csv('./result20200730.csv',header=False,index=True)
```


```python

```
