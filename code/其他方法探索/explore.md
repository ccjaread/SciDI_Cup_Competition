```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['savefig.dpi'] = 50

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler((-1,1))

import matplotlib.image as mpimg

import time

import re

from scipy import signal


from sklearn import linear_model
from scipy import stats
```


```python
import pickle
import glob
import os
from io import BytesIO
```


```python
folders=glob.glob('./AstroSet/0*')
```


```python
len(folders)
```




    26




```python
from scipy.signal import find_peaks
```

#### Basline pattern


```python
with open('./Baseline/abstar_template/flare_star.txt') as f:
    fs=f.readlines()
with open('./Baseline/abstar_template/microlensing.txt') as f:
    ml=f.readlines()
fs=[eval(i) for i in fs]
ml=[eval(i) for i in ml]
```


```python
plt.plot(fs)
plt.gca().invert_yaxis()
```


![png](output_7_0.png)



```python
plt.plot(ml)
plt.gca().invert_yaxis()
```


![png](output_8_0.png)


#### Self defined funs


```python
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
```


```python
def flatten_data(data):
    length=len(data)
    org_y=data
    xaxis=[x for x in range(length)]
    fit_x=[x for x in range(length*2)]
    fit_y=np.concatenate((org_y,org_y-org_y[0]+org_y[-1]))
    
    model = linear_model.HuberRegressor(epsilon=1.35)#HuberRegressor
    model.fit(np.array(fit_x).reshape(-1,1), fit_y)
    pred_y=model.predict(np.array(xaxis).reshape(-1,1))
    
    
    limit_times=5
    start_time=0
    while np.abs(model.coef_[0])>1e-05 and start_time<=limit_times:
        org_y=org_y-pred_y
        fit_y=np.concatenate((org_y,org_y-org_y[0]+org_y[-1]))
        model.fit(np.array(fit_x).reshape(-1,1), fit_y)
        pred_y=model.predict(np.array(xaxis).reshape(-1,1))
        start_time=start_time+1

    return org_y,pred_y,model.coef_
```


```python
def over_lap_rate(a,b):
    '''a,b are [start,end] shape 1,2'''
    if b[0]<a[0]:
        a,b=b,a
    if a[1]<=b[0]:
        return 0
    if a[1]>b[0] and a[1]<=b[1]:
        return (a[1]-b[0])/(b[1]-a[0])
    else:
        return 1
```


```python
def calc_seg_p_value(data,vline_space):
    '''calc segments p_value by ttest
    plot and return new cut_pos of segments'''
    length=len(data)
    xaxis=np.arange(length)
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(max(length//30,5),10),sharex=True)
    
    max_p=data.max() 
    #for plot annotation position only
    vline_space=vline_space.astype(int) 
    #make sure vline_space is int as it'll be used as index
    
    colors = [plt.cm.tab20(each)
          for each in np.linspace(0, 1, 20)]
    colors.extend(colors)
    colors.extend(colors)
    #gen colors 40 should be enough
    

    new_line_space=[0]
    color_sel=0
    
    #get p_values first
    p_value_lst=[]
    p_value_pos=[]
    for v_idx in range(1,len(vline_space)-1):
#         (statistic, pvalue) = stats.ttest_ind(data[vline_space[v_idx-1]:vline_space[v_idx]],\
#                                               data[vline_space[v_idx]:vline_space[v_idx+1]])
        (statistic, pvalue) = stats.mannwhitneyu(data[vline_space[v_idx-1]:vline_space[v_idx]],\
                                              data[vline_space[v_idx]:vline_space[v_idx+1]])
        
        
        
        p_value_lst.append(pvalue)
        p_value_pos.append((vline_space[v_idx]+vline_space[v_idx+1])/2)
    
    #set p_value_threshold by manual or auto
#     p_value_threshold=0.15 #manual
    p_value_threshold=max(1e-3,np.mean(p_value_lst)-3*np.std(p_value_lst)/np.sqrt(len(p_value_lst)))
#     p_value_threshold=np.percentile(p_value_lst, 5)
    
    #plot segments
    ax1.scatter(xaxis[vline_space[0]:vline_space[1]],\
                data[vline_space[0]:vline_space[1]],\
                color=colors[color_sel],alpha=0.7)
    #first draw the first segment
    #then other segments
    for v_idx in range(1,len(vline_space)-1):
        ax1.axvline(vline_space[v_idx],c='black',ls='--')
        
        pvalue=p_value_lst[v_idx-1]
        texts='{},p:{:.4f}'.format(v_idx,pvalue)
        
        if pvalue>=p_value_threshold:
            color_seg=colors[color_sel]
        else:
            color_sel=color_sel+1
            color_seg=colors[color_sel]
            new_line_space.append(vline_space[v_idx])
        
        ax1.scatter(xaxis[vline_space[v_idx]:vline_space[v_idx+1]],\
                    data[vline_space[v_idx]:vline_space[v_idx+1]],\
                    color=color_seg,alpha=0.7)
        ax1.text((vline_space[v_idx]+vline_space[v_idx+1])/2,max_p*0.9,texts,ha='center')
        
    ax2.plot(p_value_pos,p_value_lst,marker='o', linestyle='dashed')
    new_line_space.append(length-1)
    plt.show()
    return p_value_lst,np.array(new_line_space)
```


```python
def calc_seg_p_value_reverse(data,vline_space):
    '''calc segments p_value by ttest
    plot and return new cut_pos of segments'''
    length=len(data)
    xaxis=np.arange(length)
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(max(length//30,5),10),sharex=True)
    
    max_p=data.max() 
    #for plot annotation position only
    vline_space=vline_space.astype(int) 
    #make sure vline_space is int as it'll be used as index
    
    colors = [plt.cm.tab20(each)
          for each in np.linspace(0, 1, 20)]
    colors.extend(colors)
    #gen colors 40 should be enough
    

    new_line_space=[0]
    color_sel=0
    
    #get p_values first
    p_value_lst=[]
    p_value_pos=[]
    for v_idx in range(2,len(vline_space)):
#         (statistic, pvalue) = stats.ttest_ind(data[vline_space[v_idx-1]:vline_space[v_idx]],\
#                                               data[vline_space[v_idx]:vline_space[v_idx+1]])
        (statistic, pvalue) = stats.mannwhitneyu(data[vline_space[-v_idx]:vline_space[-v_idx+1]],\
                                              data[vline_space[-v_idx-1]:vline_space[-v_idx]])
        
        
        
        p_value_lst.append(pvalue)
        p_value_pos.append((vline_space[-v_idx-1]+vline_space[-v_idx])/2)
    
    #set p_value_threshold by manual or auto
#     p_value_threshold=0.15 #manual
    p_value_threshold=max(1e-3,np.mean(p_value_lst)-3*np.std(p_value_lst)/np.sqrt(len(p_value_lst)))
#     p_value_threshold=np.percentile(p_value_lst, 5)
    
    #plot segments
    ax1.scatter(xaxis[vline_space[-2]:vline_space[-1]],\
                data[vline_space[-2]:vline_space[-1]],\
                color=colors[color_sel],alpha=0.7)
    #first draw the first segment
    #then other segments
    for v_idx in range(2,len(vline_space)):
        ax1.axvline(vline_space[-v_idx],c='black',ls='--')
        
        pvalue=p_value_lst[v_idx-2]
        texts='p:{:.4f}'.format(pvalue)
        
        if pvalue>=p_value_threshold:
            color_seg=colors[color_sel]
        else:
            color_sel=color_sel+1
            color_seg=colors[color_sel]
            new_line_space.append(vline_space[-v_idx])
        
        ax1.scatter(xaxis[vline_space[-v_idx-1]:vline_space[-v_idx]],\
                    data[vline_space[-v_idx-1]:vline_space[-v_idx]],\
                    color=color_seg,alpha=0.7)
        ax1.text((vline_space[-v_idx-1]+vline_space[-v_idx])/2,max_p*0.9,texts,ha='center')
        
    ax2.plot(p_value_pos,p_value_lst,marker='o', linestyle='dashed')
    new_line_space.append(0)
    new_line_space.reverse()
    plt.show()
    return p_value_lst,np.array(new_line_space)
```


```python
def calc_seg_p_value_matrix(data,vline_space):
    '''calc segments paires p_value by ttest
    gen matrix
    plot and return new cut_pos of segments'''
    
    vline_space=vline_space.astype(int) 
    #make sure vline_space is int as it'll be used as index
    length=len(vline_space)-1
    #num of segments
    
    mat=np.zeros((length,length))
    
    for idx_row in range(length-1):
        for idx_col in range(idx_row+1,length):
            (statistic, pvalue) = stats.mannwhitneyu(data[vline_space[idx_row]:vline_space[idx_row+1]],\
                                              data[vline_space[idx_col]:vline_space[idx_col+1]])
            mat[idx_row,idx_col]=pvalue
    mat=np.mat(mat)
    mat=mat+mat.T
    fig,ax=plt.subplots()
    ax.imshow(mat,cmap='jet')
    plt.show()
#     print(mat)
```


```python

```


```python
def show_pattern_seg(data,density=100):
    length=len(data)
    print('length:{}'.format(length))
#     data,pred_y,coef_=flatten_data(data)
#     data[data>pred_y]=pred_y[data>pred_y]
        
    vline_space=np.round(np.linspace(0,length,length//density))
    
    #calc and plot
    p_value_lst,new_line_space=calc_seg_p_value(data,vline_space)
    
#     p_value_lst_2,new_line_space_2=calc_seg_p_value(data,new_line_space)
    
#     p_value_lst_3,new_line_space_3=calc_seg_p_value(data,new_line_space_2)
    
    calc_seg_p_value_matrix(data,vline_space)
#     p_value_lst_r,new_line_space_r=calc_seg_p_value_reverse(data,new_line_space_3)
        
    
    return p_value_lst
```


```python
def show_file(insp_starid):
    insp_folder=re.search(r'_(0\d{2}_\d{8}-G\d{4})_',insp_starid).group(1)
    insp_path='./AstroSet/'+insp_folder+'/'+insp_starid
    data=pd.read_table(insp_path,sep=u' ',names=['time','data','dev'])
    data.sort_values(by='time',ascending=True,inplace=True)
    breakpoints=get_breakpoints(data)
    show_breakpoints(data,breakpoints)
#     data['filted']=0
    whole_data=np.array([0])
#     whole_dev=np.array([0])
    for i in range(len(breakpoints)-1):
        new_data=data.loc[breakpoints[i]:breakpoints[i+1]-1,'data'].values
        gap=np.median(whole_data)-np.median(new_data)
        new_data=new_data+gap
        whole_data=np.concatenate((whole_data,new_data))
#         whole_dev=np.concatenate((whole_data,data.loc[breakpoints[i]:breakpoints[i+1]-1,'dev'].values))
#         print('from {} to {},{} to {} for breakpoints'.format(breakpoints[i],breakpoints[i+1]-1,i,i+1))
#         if breakpoints[i+1]-1-breakpoints[i]<100:
#             print('Too small data set to process, ignored\n\n')
#         else:
# #             data.loc[breakpoints[i]:breakpoints[i+1]-1,'filted']= signal.filtfilt(b, a, data.loc[breakpoints[i]:breakpoints[i+1]-1,'data'].values)  #data为要过滤的信号
    show_pattern_seg(whole_data)
#     show_pattern_seg(whole_dev)
```

#### Explore Data


```python
train=pd.read_csv('./AstroSet/train_abnormal',names=['starid','tag','path'])
# train_lens=train.loc[train.tag=='microlensing'].copy()
# train_lens.reset_index(inplace=True)
# train_flare=train.loc[train.tag=='flare star'].copy()
# train_flare.reset_index(inplace=True)

# i=17
# print(train.tag[i])
# show_file(train.starid[i])
```


```python
train.head()
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
      <th>tag</th>
      <th>path</th>
      <th>full_path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ref_021_18450425-G0014_703997_4523</td>
      <td>flare star</td>
      <td>AstroSet/021_18450425-G0014</td>
      <td>AstroSet/021_18450425-G0014\ref_021_18450425-G...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ref_031_12010765-G0013_520170_227006</td>
      <td>microlensing</td>
      <td>AstroSet/031_12010765-G0013</td>
      <td>AstroSet/031_12010765-G0013\ref_031_12010765-G...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ref_021_24310595-G0013_746468_14893</td>
      <td>flare star</td>
      <td>AstroSet/021_24310595-G0014</td>
      <td>AstroSet/021_24310595-G0014\ref_021_24310595-G...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ref_022_06300085-G0013_1548728_32012</td>
      <td>flare star</td>
      <td>AstroSet/022_06300085-G0013</td>
      <td>AstroSet/022_06300085-G0013\ref_022_06300085-G...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ref_022_15730595-G0013_391462_6330</td>
      <td>flare star</td>
      <td>AstroSet/022_15730595-G0013</td>
      <td>AstroSet/022_15730595-G0013\ref_022_15730595-G...</td>
    </tr>
  </tbody>
</table>
</div>




```python
train['full_path']=train[['path','starid']].apply(lambda x:x['path']+'\\'+x['starid'],axis=1)
```


```python
#special case record
#starid[8] is a special case
#1
```


```python
insp_path=starpath[14]
```


```python
def get_whole_data(insp_path):
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
    return whole_data,breakpoints
```


```python
# insp_starid=train.starid[15]
# insp_folder=re.search(r'_(0\d{2}_\d{8}-G\d{4})_',insp_starid).group(1)
# insp_path='./AstroSet/'+insp_folder+'/'+insp_starid
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
#         whole_dev=np.concatenate((whole_data,data.loc[breakpoints[i]:breakpoints[i+1]-1,'dev'].values))
#         print('from {} to {},{} to {} for breakpoints'.format(breakpoints[i],breakpoints[i+1]-1,i,i+1))
#         if breakpoints[i+1]-1-breakpoints[i]<100:
#             print('Too small data set to process, ignored\n\n')
#         else:
# #             data.loc[breakpoints[i]:breakpoints[i+1]-1,'filted']= signal.filtfilt(b, a, data.loc[breakpoints[i]:breakpoints[i+1]-1,'data'].values)  #data为要过滤的信号
p_value_lst=show_pattern_seg(whole_data,density=100)
#     show_pattern_seg(whole_dev)
```

    length:1565
    


![png](output_26_1.png)



![png](output_26_2.png)



```python
import glob
```


```python
starpath=glob.glob('./AstroSet/021_14110255-G0013/ref*')
```


```python
len(starpath)
```




    19088




```python
b, a = signal.butter(8, 0.1, 'lowpass')   #配置滤波器 8 表示滤波器的阶数
```


```python
insp_path=starpath[9]
```


```python
idx=5
insp_path=train.path[idx]+'\\'+train.starid[idx]
```


```python
whole_data=get_whole_data(insp_path)
whole_data=-whole_data
```


```python
sos = signal.butter(10, 10, 'lp', fs=1000, output='sos')
```


```python
whole_data = signal.sosfilt(sos, whole_data)
```


```python
# whole_data= signal.filtfilt(b, a, whole_data)
```


```python
# peaks, _ = signal.find_peaks(whole_data, height=0)
plt.figure(figsize=(len(whole_data)//30,5))
plt.plot(whole_data,'o',alpha=0.5)
# plt.plot(peaks, whole_data[peaks], "o")
plt.plot(np.zeros_like(whole_data), "--", color="gray")
plt.show()
```


![png](output_37_0.png)



```python
from sklearn.cluster import DBSCAN


# #############################################################################
# Generate sample data
data_size=len(whole_data)

X=np.concatenate((np.arange(data_size),whole_data),axis=0).reshape(-1,data_size).T

X = preprocessing.StandardScaler().fit_transform(X)

# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.5, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
```

    Estimated number of clusters: 1
    Estimated number of noise points: 13
    


```python
# #############################################################################
# Plot result
import matplotlib.pyplot as plt
plt.figure(figsize=(len(whole_data)//30,5))

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
```


![png](output_39_0.png)



```python
x=np.arange(data_size)
y=whole_data
a=np.polyfit(x,y,3)#用2次多项式拟合x，y数组
b=np.poly1d(a)#拟合完之后用这个函数来生成多项式对象
c=b(x)#生成多项式对象之后，就是获取x在这个多项式处的值
plt.figure(figsize=(len(whole_data)//30,5))
plt.scatter(x,y,marker='o',label='original datas')#对原始数据画散点图
plt.plot(x,c,ls='--',c='red',label='fitting with second-degree polynomial')#对拟合之后的数据，也就是x，c数组画图
plt.legend()
plt.show()
```


![png](output_40_0.png)



```python
whole_data=get_whole_data(insp_path)
whole_data=-whole_data
```


```python
whole_data
```




    array([-0.  ,  0.13,  0.12, ...,  0.14,  0.15,  0.23])




```python
def get_all_fitted(data,density=50):
    length=len(data)
#     print('length:{}'.format(length))
#     data,pred_y,coef_=flatten_data(data)
#     data[data>pred_y]=pred_y[data>pred_y]
#     vline_space=np.round(np.linspace(0,length,length//density))
    vline_space, _ = signal.find_peaks(data, distance=density)
    
    vline_space=vline_space.astype(int) 
    #make sure vline_space is int as it'll be used as index
    fitted_data=calc_polyfit(data[:vline_space[0]])
#     fitted_data=np.array([0])
    for v_idx in range(0,len(vline_space)-1):
        fitted_data_tmp=calc_polyfit(data[vline_space[v_idx]:vline_space[v_idx+1]])
#         gap=np.median(fited_data)-np.median(fited_data_tmp)
#         fited_data_tmp=fited_data_tmp+gap
        fitted_data=np.concatenate((fitted_data,fitted_data_tmp))
        
    
    return fitted_data,vline_space
```


```python
def calc_polyfit(data):
    
    length=len(data)
    x=np.arange(length)
    
    model = linear_model.HuberRegressor(epsilon=1.35)#HuberRegressor
    model.fit(x.reshape(-1,1), data)
    c=model.predict(x.reshape(-1,1))
    score=model.score(x.reshape(-1,1), data)
    c=c*score
#     a=np.polyfit(x,data,1)#用2次多项式拟合x，y数组
#     b=np.poly1d(a)#拟合完之后用这个函数来生成多项式对象
#     c=b(x)#生成多项式对象之后，就是获取x在这个多项式处的值
    
    return c
```


```python
insp_path=starpath[9]
```


```python
idx=16
insp_path=train.path[idx]+'\\'+train.starid[idx]
```


```python
insp_path.split('\\')
```




    ['AstroSet/024_13500085-G0013', 'ref_024_13500085-G0013_1397642_27534']




```python
def find_event(insp_path,threshold=25,density=50):
    whole_data,breakpoints=get_whole_data(insp_path)
    whole_data=-whole_data
    # whole_data=scaler.fit_transform(whole_data.reshape(-1,1)).reshape(1,-1)[0]

    whole_data = preprocessing.StandardScaler().fit_transform(whole_data.reshape(-1,1)).flatten()

    whole_data_fitted,vline_space=get_all_fitted(whole_data,density)

    # #plot orginal data
    plt.figure(figsize=(len(whole_data)//30,5))
    plt.plot(whole_data,'o')
    plt.plot(vline_space, whole_data[vline_space], "o",c='r',ms=20)
    
    for pnt in breakpoints:
        plt.axvline(pnt,c='black',ls='--')

    plt.plot(whole_data_fitted,'o',alpha=0.5)
    # # #plot fitted data

    #check if the right pattern
    chk_model = linear_model.HuberRegressor(epsilon=1.35)#HuberRegressor
    chk_model.fit(np.arange(len(whole_data_fitted)).reshape(-1,1), whole_data_fitted)
    avg_line=chk_model.predict(np.arange(len(whole_data_fitted)).reshape(-1,1))
    whole_data_fitted=whole_data_fitted-avg_line
    flag=whole_data_fitted[whole_data_fitted>0].sum()-np.fabs(whole_data_fitted[whole_data_fitted<0]).sum()
#     print(flag)
    if flag>threshold:
    #plot reslut
        
        center_pos=np.argmin(np.fabs(vline_space-np.argmax(whole_data_fitted)))
        left_pos=max(0,center_pos-3)
        right_pos=min(center_pos+3,len(vline_space)-1)
        real_max_pos=np.argmax(whole_data[vline_space[left_pos]:vline_space[right_pos]])+vline_space[left_pos]
        real_left_pos=int(real_max_pos-2*density)
        real_right_pos=int(real_max_pos+2*density)
        #plot whole
#         plt.figure(figsize=(len(whole_data)//30,5))
#         plt.plot(whole_data,'o')
#         plt.plot(vline_space, whole_data[vline_space], "o",c='r',ms=20)
#         plt.axvline(real_max_pos,c='b')
        plt.fill_betweenx(np.linspace(-1,1,10),real_left_pos,real_right_pos,color='r',alpha=0.5)
        
        media_pos=(real_right_pos+real_left_pos)/2
        gapp=real_right_pos-real_left_pos
        plt.fill_betweenx(np.linspace(-1,1,10),media_pos-gapp*0.015,media_pos+gapp*0.015,color='yellow',alpha=0.5)
        plt.axvline(media_pos,c='red',ls='--')
#         plt.show()
        
#         fig,ax=plt.subplots(figsize=(1,1))

#         ax.scatter(np.arange(real_right_pos-real_left_pos),whole_data[real_left_pos:real_right_pos],color='black',alpha=0.7,s=3)
#         ax.axis('off')
#         ax.margins(0, 0)
        
#         res_path = r'./Train2/'
#         fname=insp_path.split('\\')[-1]
#         seg='-'.join([str(real_left_pos),str(real_right_pos)])
#         fig.savefig(res_path+fname+'+'+seg+'.png')
#         plt.close(fig)
        
#         #in memory
#         buffer_ = BytesIO()
#         plt.savefig(buffer_,format = 'png')
#         plt.close(fig)
#         buffer_.seek(0)
#         #从内存中读取
#         img_tmp=img_to_array(buffer_)
#         #释放缓存    
#         buffer_.close()
#         #in memory
        return True
    return False
```


```python
for idx in range(len(train)):
    insp_path=train.path[idx]+'\\'+train.starid[idx]
    find_event(insp_path,threshold=25,density=50)
```


![png](output_49_0.png)



![png](output_49_1.png)



![png](output_49_2.png)



![png](output_49_3.png)



![png](output_49_4.png)



![png](output_49_5.png)



![png](output_49_6.png)



![png](output_49_7.png)



![png](output_49_8.png)



![png](output_49_9.png)



![png](output_49_10.png)



![png](output_49_11.png)



![png](output_49_12.png)



![png](output_49_13.png)



![png](output_49_14.png)



![png](output_49_15.png)



![png](output_49_16.png)



![png](output_49_17.png)



```python
for idx in range(len(train)):
    insp_path=train.path[idx]+'\\'+train.starid[idx]
    find_event(insp_path,threshold=25,density=50)
```


```python
#ref_021_14110255-G0013_324851_13815+1105-1305脉冲
```


```python
insp_starid='ref_021_14110255-G0013_324851_13815'
insp_folder=re.search(r'_(0\d{2}_\d{8}-G\d{4})_',insp_starid).group(1)
insp_path='./AstroSet/'+insp_folder+'/'+insp_starid

find_event(insp_path)
```




    True




![png](output_52_1.png)



```python
with open('./starid_path_dict.pickle','rb') as f:
    starid_path_dict=pickle.load(f)
```


```python
insp_path
```




    './AstroSet/021_18450425-G0014/ref_021_18450425-G0014_357415_19467'




```python
insp_starid='ref_034_21640255-G0013_569111_13375'
# insp_folder=re.search(r'_(0\d{2}_\d{8}-G\d{4})_',insp_starid).group(1)
# insp_path='./AstroSet/'+insp_folder+'/'+insp_starid
# insp_path=r'C:\Users\CJA_HOME\Jupyter_Data\SciDI_Cup\AstroSet\021_24310595-G0014\ref_021_24310595-G0013_734569_20438'
insp_path=starid_path_dict_2.get(insp_starid)
find_event(insp_path)
```




    True




![png](output_55_1.png)



```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
```


```python
def autocorrelation(x,lags):#计算lags阶以内的自相关系数，返回lags个值，分别计算序列均值，标准差
    n = len(x)
    x = np.array(x)
    result = [np.correlate(x[i:]-x[i:].mean(),x[:n-i]-x[:n-i].mean())[0]/(x[i:].std()*x[:n-i].std()*(n-i)) for i in range(1,lags+1)]
    result = np.array(result)
#     diff_=result[:-1]-result[1:]
#     return diff_.mean()
    return result[-1]
```


```python
whole_data=get_whole_data(insp_path)
whole_data=-whole_data
# plot_acf(whole_data)
```


```python
autocorrelation(whole_data,5)
```




    0.8096800124006339




```python
import glob
```


```python
find_lst=glob.glob('./Finded_0906/*.png')
```


```python
len(find_lst)
```




    13793




```python
find_lst[0].split('\\')[-1].split('+')[0]
```




    'ref_021_14110255-G0013_314601_24669'




```python
re.search(r'(ref_\d{3}_.+-G\d{4})',find_lst[0].split('\\')[-1].split('+')[0]).group(1)
```




    'ref_021_14110255-G0013'




```python
find_folder=[]
for find_file in find_lst:
    find_folder.append(re.search(r'(ref_\d{3}_.+-G\d{4})',find_file.split('\\')[-1].split('+')[0]).group(1))
```


```python
def find_name(x):
    return re.search(r'(ref_.*)\+\d',x).group(1)
```


```python
find_starid=set([find_name(x) for x in find_lst])

```


```python
find_starid_train=set([find_name(x) for x in glob.glob('./Train93/*.png')])
```


```python
len(find_starid_train)
```




    15




```python
len(find_starid)
```




    13793




```python
set(train.starid.to_list())
```




    {'ref_021_18450425-G0014_703997_4523',
     'ref_021_24310595-G0013_746468_14893',
     'ref_022_06300085-G0013_1548728_32012',
     'ref_022_15730595-G0013_391462_6330',
     'ref_022_15730595-G0013_481532_2590',
     'ref_023_15730595-G0013_378567_3040',
     'ref_024_12230255-G0013_303557_17479',
     'ref_024_12230255-G0013_393825_55632',
     'ref_024_13500085-G0013_1397642_27534',
     'ref_031_12010765-G0013_520170_227006',
     'ref_033_04500085-G0013_1485992_22874',
     'ref_033_04700255-G0013_25519_18341',
     'ref_033_15300085-G0013_1742006_16050',
     'ref_033_16810765-G0013_482792_15702',
     'ref_041_14110425-G0013_422384_26026',
     'ref_043_04500085-G0013_2358479_2504',
     'ref_043_18590595-G0013_709318_8662',
     'ref_044_16280425-G0013_364820_9174'}




```python
find_starid_train.intersection(set(train.starid.to_list()))
```




    {'ref_021_18450425-G0014_703997_4523',
     'ref_021_24310595-G0013_746468_14893',
     'ref_022_06300085-G0013_1548728_32012',
     'ref_022_15730595-G0013_391462_6330',
     'ref_022_15730595-G0013_481532_2590',
     'ref_023_15730595-G0013_378567_3040',
     'ref_024_12230255-G0013_303557_17479',
     'ref_024_13500085-G0013_1397642_27534',
     'ref_033_04500085-G0013_1485992_22874',
     'ref_033_04700255-G0013_25519_18341',
     'ref_033_15300085-G0013_1742006_16050',
     'ref_033_16810765-G0013_482792_15702',
     'ref_041_14110425-G0013_422384_26026',
     'ref_043_04500085-G0013_2358479_2504',
     'ref_044_16280425-G0013_364820_9174'}




```python
find_starid.intersection(set(train.starid.to_list()))
```




    {'ref_021_18450425-G0014_703997_4523',
     'ref_021_24310595-G0013_746468_14893',
     'ref_022_06300085-G0013_1548728_32012',
     'ref_022_15730595-G0013_391462_6330',
     'ref_022_15730595-G0013_481532_2590',
     'ref_023_15730595-G0013_378567_3040',
     'ref_024_12230255-G0013_303557_17479',
     'ref_024_13500085-G0013_1397642_27534'}




```python
insp_starid_lst=['ref_021_24960425-G0014_641731_14386',
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
```


```python
res9=pd.read_csv('./res_20200909-2.csv',names=['starid','l','r','event'])
res12=pd.read_csv('./res_20200912.csv',names=['starid','l','r','event'])
```


```python
for insp_starid in insp_starid_lst:
    insp_path=starid_path_dict_2.get(insp_starid)
    find_event(insp_path)
```

    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\numpy\core\fromnumeric.py:3257: RuntimeWarning: Mean of empty slice.
      out=out, **kwargs)
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\numpy\core\_methods.py:161: RuntimeWarning: invalid value encountered in double_scalars
      ret = ret.dtype.type(ret / rcount)
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    


![png](output_76_1.png)



![png](output_76_2.png)



![png](output_76_3.png)



![png](output_76_4.png)



![png](output_76_5.png)



![png](output_76_6.png)



![png](output_76_7.png)



![png](output_76_8.png)



![png](output_76_9.png)



![png](output_76_10.png)



![png](output_76_11.png)



![png](output_76_12.png)



![png](output_76_13.png)



![png](output_76_14.png)



![png](output_76_15.png)



![png](output_76_16.png)



![png](output_76_17.png)



![png](output_76_18.png)



![png](output_76_19.png)



![png](output_76_20.png)



![png](output_76_21.png)



![png](output_76_22.png)



![png](output_76_23.png)



![png](output_76_24.png)



![png](output_76_25.png)



![png](output_76_26.png)



![png](output_76_27.png)



![png](output_76_28.png)



![png](output_76_29.png)



![png](output_76_30.png)



![png](output_76_31.png)



![png](output_76_32.png)



![png](output_76_33.png)



![png](output_76_34.png)



```python
for insp_starid in res9.starid.to_list():
    insp_path=starid_path_dict_2.get(insp_starid)
    find_event(insp_path)
```

    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\numpy\core\fromnumeric.py:3257: RuntimeWarning: Mean of empty slice.
      out=out, **kwargs)
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\numpy\core\_methods.py:161: RuntimeWarning: invalid value encountered in double_scalars
      ret = ret.dtype.type(ret / rcount)
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    C:\Users\CJA_HOME\WPy64-3760\python-3.7.6.amd64\lib\site-packages\ipykernel_launcher.py:11: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      # This is added back by InteractiveShellApp.init_path()
    


![png](output_77_1.png)



![png](output_77_2.png)



![png](output_77_3.png)



![png](output_77_4.png)



![png](output_77_5.png)



![png](output_77_6.png)



![png](output_77_7.png)



![png](output_77_8.png)



![png](output_77_9.png)



![png](output_77_10.png)



![png](output_77_11.png)



![png](output_77_12.png)



![png](output_77_13.png)



![png](output_77_14.png)



![png](output_77_15.png)



![png](output_77_16.png)



![png](output_77_17.png)



![png](output_77_18.png)



![png](output_77_19.png)



![png](output_77_20.png)



![png](output_77_21.png)



![png](output_77_22.png)



![png](output_77_23.png)



![png](output_77_24.png)



![png](output_77_25.png)



![png](output_77_26.png)



![png](output_77_27.png)



![png](output_77_28.png)



![png](output_77_29.png)



![png](output_77_30.png)



![png](output_77_31.png)



![png](output_77_32.png)



![png](output_77_33.png)



![png](output_77_34.png)



![png](output_77_35.png)



![png](output_77_36.png)



![png](output_77_37.png)



![png](output_77_38.png)



![png](output_77_39.png)



![png](output_77_40.png)



![png](output_77_41.png)



![png](output_77_42.png)



![png](output_77_43.png)



![png](output_77_44.png)



![png](output_77_45.png)



![png](output_77_46.png)



![png](output_77_47.png)



![png](output_77_48.png)



![png](output_77_49.png)



![png](output_77_50.png)



```python

```


```python
insp_starid_lst=['ref_034_21640255-G0013_569111_13375',
 'ref_022_13500085-G0014_288288_19986',
 'ref_043_12010765-G0013_497481_81201',
 'ref_043_06300085-G0013_1469081_18682',
 'ref_044_02700085-G0013_1154500_44491']
```


```python
for insp_starid in insp_starid_lst:
    insp_path=starid_path_dict_2.get(insp_starid)
    find_event(insp_path)
```


![png](output_80_0.png)



![png](output_80_1.png)



![png](output_80_2.png)



![png](output_80_3.png)



![png](output_80_4.png)



```python

```


```python

```


```python

```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import matplotlib.image as mpimg
```


```python
def img_to_array(imageFile):
    img = mpimg.imread(imageFile).astype(np.float)
    img=img[:,:,0]
    img=1-img
    return img
```


```python
img1.shape
```




    (50, 50)




```python
np.concatenate((np.arange(10),np.arange(10),np.arange(10)))
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1,
           2, 3, 4, 5, 6, 7, 8, 9])




```python
def compare_img(img_target,img_tmp):
    res_lst=[]
    expand_img=np.concatenate((np.zeros((50,5)),img_target,np.zeros((50,5))),axis=1)
    for shift_val in range(10):
        shift_data=expand_img[:,shift_val:shift_val+50]
        diff=shift_data-img_tmp
        diff[diff!=0]=1
        res_lst.append(diff.sum())
    return min(res_lst)
```


```python
%%time
compare_img(img2,img3)
```

    Wall time: 1.03 ms
    




    837.0




```python
img1=img_to_array('./Train92/1/ref_021_18450425-G0014_703997_4523+1900-2100.png')
img2=img_to_array('./Train92/1/ref_021_24310595-G0013_746468_14893+1019-1219.png')
img3=img_to_array('./Train92/2/ref_022_15730595-G0013_391462_6330+2057-2257.png')
img4=img_to_array('./Train92/0/ref_021_14110255-G0013_319570_39575+1234-1434.png')
```


```python
expand_img=np.concatenate((np.zeros((50,5)),img1,np.zeros((50,5))),axis=1)
plt.imshow(expand_img[:,5:55])
```




    <matplotlib.image.AxesImage at 0x20e86071188>




![png](output_91_1.png)



```python
plt.imshow(img4-img1)
```




    <matplotlib.image.AxesImage at 0x20e85e3e648>




![png](output_92_1.png)



```python

```


```python
plt.imshow(img1)
```




    <matplotlib.image.AxesImage at 0x20e87a2b8c8>




![png](output_94_1.png)



```python
img1[img1]
```




    0.00392156862745098




```python
plt.imshow(img2+img4)
```




    <matplotlib.image.AxesImage at 0x20e85b45ec8>




![png](output_96_1.png)



```python
plt.imshow(img2)
```




    <matplotlib.image.AxesImage at 0x20e89792488>




![png](output_97_1.png)



```python
plt.imshow(img3)
```




    <matplotlib.image.AxesImage at 0x20e896f2d88>




![png](output_98_1.png)



```python
plt.imshow(img4)
```




    <matplotlib.image.AxesImage at 0x20e87bc55c8>




![png](output_99_1.png)



```python
img1.sum(axis=0)
```




    array([0.19607843, 0.19607843, 0.19607843, 0.19607843, 0.19607843,
           0.19607843, 0.17866974, 0.16745867, 0.16358324, 0.16575163,
           0.17334871, 0.17168781, 0.16552095, 0.16964245, 0.17208766,
           0.16878124, 0.17124183, 0.17122645, 0.16492118, 0.16589004,
           0.16545944, 0.17347174, 0.17611688, 0.17551711, 0.163614  ,
           0.15826221, 0.16367551, 0.16196847, 0.16615148, 0.16916571,
           0.17430219, 0.17519416, 0.17301038, 0.17617839, 0.16795079,
           0.17051903, 0.17265667, 0.16859669, 0.16678201, 0.17405613,
           0.1730719 , 0.17038062, 0.16842753, 0.16286044, 0.17202614,
           0.18105344, 0.19607843, 0.19607843, 0.19607843, 0.19607843])




```python
plt.plot(img1.sum(axis=1))
```




    [<matplotlib.lines.Line2D at 0x20e877e68c8>]




![png](output_101_1.png)



```python
plt.plot(img2.sum(axis=1))
```




    [<matplotlib.lines.Line2D at 0x20e87618e08>]




![png](output_102_1.png)



```python
plt.plot(img3.sum(axis=1))
```




    [<matplotlib.lines.Line2D at 0x20e8758cf88>]




![png](output_103_1.png)



```python
plt.plot(img4.sum(axis=1))
```




    [<matplotlib.lines.Line2D at 0x20e87997e88>]




![png](output_104_1.png)



```python
sim, path=fastdtw(img4.sum(axis=1), img3.sum(axis=1), dist=euclidean)
sim
```




    0.4774317520323628




```python
def calc_similarity(tgt,fs,ml):
#     tgt=squeeze_tgt(tgt, 60)
    tgt=scaler.fit_transform(tgt.reshape(-1,1)).reshape(1,-1)[0]
    sim_ml, path = fastdtw(tgt, ml, dist=euclidean)
    sim_fs, path = fastdtw(tgt, fs, dist=euclidean)
    if sim_fs<=sim_ml:
        sim=sim_fs
        res='flare star'
    else:
        sim=sim_ml
        res='microlensing'
    sim=sim/60
    return res,sim
```


```python
stars=glob.glob('./AstroSet/0*/ref*')
```


```python
stars[:5]
```




    ['./AstroSet\\021_24960425-G0014\\ref_021_24960425-G0014_617207_5180',
     './AstroSet\\021_24960425-G0014\\ref_021_24960425-G0014_617207_5241',
     './AstroSet\\021_24960425-G0014\\ref_021_24960425-G0014_617209_6086',
     './AstroSet\\021_24960425-G0014\\ref_021_24960425-G0014_617210_6397',
     './AstroSet\\021_24960425-G0014\\ref_021_24960425-G0014_617211_6217']




```python
len(stars)
```




    933889




```python
stars[0].split('\\')[-1]
```




    'ref_021_24960425-G0014_617207_5180'




```python
starid_path_dict_2={}
for i in range(len(stars)):
    starid_path_dict_2[stars[i].split('\\')[-1]]=stars[i]
```


```python
starid_path_dict_2['ref_021_24960425-G0014_617207_5180']
```




    './AstroSet\\021_24960425-G0014\\ref_021_24960425-G0014_617207_5180'




```python
with open('./starid_path_dict2.pickle','wb') as f:
    pickle.dump(starid_path_dict_2,f)
```


```python

```
