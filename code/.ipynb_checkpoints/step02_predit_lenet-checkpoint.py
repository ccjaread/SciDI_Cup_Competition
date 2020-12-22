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

#生成starid 到对应数据文档路径的映射字典,这部分首次运行需按实际运行目录重新生成并替换提供的文件####
# with open('./file_lst.pickle','rb') as f:
#     files_lst=pickle.load(f)

files_lst=glob.glob(r'./AstroSet/*/ref*')
starid_path_dict={}
for file_path in files_lst:
    starid_path_dict[file_path.split('/')[-1]]=file_path

# with open('../user_data/starid_path_dict2.pickle','wb') as f:
#     pickle.dump(starid_path_dict,f)

#如果按文件夹结构组织星体文件
##starid_path_dict 初赛的starid 到对应数据文档路径的映射字典
##starid_path_dict2 复赛的starid 到对应数据文档路径的映射字典
#如果自己生成过需要与上面的文件命名一致

with open('../user_data/starid_path_dict2.pickle','rb') as f:
    starid_path_dict=pickle.load(f)
    
########################################################################

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


if __name__=='__main__':
    path_pred=r'../user_data/finded_result/*.png'

    pool_pred=glob.glob(path_pred)

    print('读取到候选天体事件{}个'.format(len(pool_pred)))

    #读取训练的模型，模型迭代好几版了，各有优缺点，默认202009122版，注意tensorflow版本，亦可重新训练下
    from tensorflow.keras.models import load_model
    model = load_model("../user_data/trained_model/LeNet_202009122_img_50_50.h5")

    model.summary()

    pool_pred_starids=set([name for name in map(find_name,pool_pred)])

    x_size=len(pool_pred)
    x = np.zeros((x_size, 50,50,1))
    #train sample only
    for i in range(x_size):
        x[i] = img_to_array(pool_pred[i])

    pred_y=model.predict(x)

    y=np.apply_along_axis(np.argmax,1,pred_y)

    score=np.apply_along_axis(np.max,1,pred_y)

    predicts=pd.DataFrame({'starid':[name for name in map(find_name,pool_pred)],\
                           'event':y,\
                           'score':score,\
                           'path':pool_pred})

    final_predicts=predicts.loc[predicts.event!=0].sort_values(by='score',ascending=False).copy()

    #开始计算除softmax外的其他筛选参数及提取片段相对和绝对位置
    #get real time 获得真实天文时间
    final_predicts['l_pos']='N'
    final_predicts['r_pos']='N'
    final_predicts[['l_pos','r_pos']]=final_predicts.apply(get_real_pos,axis=1,result_type='expand')

    #get rel idx 获得数据片段相对位置
    final_predicts['l_pos_rel']='N'
    final_predicts['r_pos_rel']='N'
    final_predicts[['l_pos_rel','r_pos_rel']]=final_predicts.apply(get_rel_pos,axis=1,result_type='expand')

    #计算5阶自回归系数，以排除一部分周期性的脉冲星，看情况使用作为筛选条件，阶数可在前面函数定义时定义lag
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

    ## 20200912提交的成绩是这条,使用LeNet_202009122_img_50_50.h5
    final_predicts.loc[(final_predicts.auto_corr<0.8) & \
                       (final_predicts.score>0.5) & \
                       (final_predicts.is_overlap_bp=='N'),\
                       ['starid','l_pos', 'r_pos','event']].to_csv('../prediction_result/res_20200912_py.csv',header=False,index=False)
    print('done')