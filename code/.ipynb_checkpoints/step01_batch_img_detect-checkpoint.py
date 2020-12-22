import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['savefig.dpi'] = 50

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler((-1,1))
# import matplotlib.image as mpimg

from scipy import signal
from sklearn import linear_model

import pickle
import glob
import os
from io import BytesIO

import multiprocessing as mp



def get_breakpoints(data):
    '''探测原始数据因不同时间段测量的断点，并输出作为下一步拼接成完整连续数据的依据'''
    diff=data.time.diff()
    diff.fillna(method='bfill',inplace=True)
    breakpoints,=np.where(diff.values>diff.mean()*4)
    breakpoints=np.append(breakpoints,len(data))
    breakpoints=np.append(0,breakpoints)
    return breakpoints

def get_whole_data(insp_path):
    '''拼接完整数据'''
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

def calc_polyfit(data):
    '''分段拟合'''
    
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

def get_all_fitted(data,density=50):
    '''分段拟合完整数据，得到抽象的趋势'''
    length=len(data)
#     print('length:{}'.format(length))
#     data,pred_y,coef_=flatten_data(data)
#     data[data>pred_y]=pred_y[data>pred_y]
#     vline_space=np.round(np.linspace(0,length,length//density))
    vline_space, _ = signal.find_peaks(data, distance=density)
    
    vline_space=vline_space.astype(int) 
    #make sure vline_space is int as it'll be used as index
#     fitted_data=calc_polyfit(data[:vline_space[0]])

    fitted_data=np.array([0])
    for v_idx in range(0,len(vline_space)-1):
        fitted_data_tmp=calc_polyfit(data[vline_space[v_idx]:vline_space[v_idx+1]])
#         gap=np.median(fited_data)-np.median(fited_data_tmp)
#         fited_data_tmp=fited_data_tmp+gap
        fitted_data=np.concatenate((fitted_data,fitted_data_tmp))
        
    
    return fitted_data,vline_space

def find_event(insp_path,threshold=25,density=50):
    '''
    寻找符合条件的事件候选，并截取一段生成50*50的图片，
    阈值threshold可以调整，越大越严格，density为限制截取时间段的参数
    res_path = r'./Finded_0909/'为图片输出地址设置，按目前要求设置为r'./user_data/finded_result/'
    '''
    try:
        whole_data=get_whole_data(insp_path)
        whole_data=-whole_data
        # whole_data=scaler.fit_transform(whole_data.reshape(-1,1)).reshape(1,-1)[0]

        whole_data = preprocessing.StandardScaler().fit_transform(whole_data.reshape(-1,1)).flatten()

        whole_data_fitted,vline_space=get_all_fitted(whole_data,density)

    #     # #plot orginal data
    #     plt.figure(figsize=(len(whole_data)//30,5))
    #     plt.plot(whole_data,'o')
    #     plt.plot(vline_space, whole_data[vline_space], "o",c='r',ms=20)

    #     plt.plot(whole_data_fitted,'o',alpha=0.5)
    #     # # #plot fitted data

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
    #         plt.fill_betweenx(np.linspace(-1,1,10),real_left_pos,real_right_pos,color='r',alpha=0.5)
    #         plt.show()

            fig,ax=plt.subplots(figsize=(1,1))

            ax.scatter(np.arange(real_right_pos-real_left_pos),whole_data[real_left_pos:real_right_pos],color='black',alpha=0.7,s=3)
            ax.axis('off')
            ax.margins(0, 0)

            res_path = r'./user_data/finded_result/
            fname=insp_path.split('\\')[-1]
            seg='-'.join([str(real_left_pos),str(real_right_pos)])
            fig.savefig(res_path+fname+'+'+seg+'.png')
            plt.close(fig)

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
    except:
        pass
    return False

#batch_detect
if __name__=='__main__':
    print('start')
    pool = mp.Pool(3)
    # 按3进程运行一方面加数，一方面减少python内存泄漏的发生
    folders=glob.glob('./AstroSet/0*')
    # 这边的目录在实际运行时需要修改
    for folder in folders[40:]:
        print(folder)
        stars=glob.glob(folder+r'/ref*')
        print(len(stars))
        pool.map(find_event,stars)
        print('done'+folder)
    pool.close()
    pool.join()
    print('all done')