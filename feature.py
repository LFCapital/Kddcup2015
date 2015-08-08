from __future__ import division
import pandas as pd
from pandas.tseries.offsets import *
import numpy as np
from datetime import datetime, date, time

# Read data
enrollment_train_df = pd.read_csv('D://KDD_RAW_DATA//train//enrollment_train.csv')
log_train_df = pd.read_csv('D://KDD_RAW_DATA//train//log_train.csv')
enrollment_test_df = pd.read_csv('D://KDD_RAW_DATA//test//enrollment_test.csv')
log_test_df = pd.read_csv('D://KDD_RAW_DATA//test//log_test.csv')

#Get the date info
log_train_df["YMD"] = pd.to_datetime(log_train_df["time"].map(lambda s: s[:10]))
log_test_df["YMD"] = pd.to_datetime(log_test_df["time"].map(lambda s: s[:10]))

# store the col names
col= ['enrollment_id','day_num','start_day','end_day','log_duration','day_ratio','duration_ratio',
        'close_num','avg_close','avg_close2','nav_num','avg_nav','avg_nav2','acc_num','avg_acc','avg_acc2','wiki_num','avg_wiki','avg_wiki2','dis_num','avg_dis','avg_dis2',
        'operation_num','avg_operation','sd_opearation','max_operation','min_operation',
        'mid_operation' ,'q25_operation','q75_operation','max_operation_ratio',
        'uniq_object_num','object_num','avg_object','sd_object','max_object',
        'min_object','mid_object' ,'q25_object','q75_object','max_object_ratio',
        'wproblem_num' ,'avg_wproblem','sd_wproblem','max_wproblem','min_wproblem',
        'mid_wproblem','q25_wproblem','q75_wproblem' ,'max_wproblem_ratio','prob_day_ratio','avg_wproblem2','avg_wproblem3',
        'avg_rproblem','sd_rproblem','max_rproblem','min_rproblem',
        'mid_rproblem','q25_rproblem','q75_rproblem','max_rproblem_ratio','avg_rproblem2','avg_rproblem3',
        'wvideo_num','avg_wvideo','sd_wvideo','max_wvideo','min_wvideo',
        'mid_wvideo','q25_wvideo','q75_wvideo','max_wvideo_ratio','video_day_ratio','avg_wvideo2','avg_wvideo3', 
        'rvideo_num', 'avg_rvideo','sd_rvideo','max_rvideo','min_rvideo','mid_rvideo',
        'q25_rvideo','q75_rvideo','max_rvideo_ratio','avg_rproblem2','avg_rproblem3',       
        'w1_log','w2_log','w3_log','w4_log' ,'flag1_2','flag3_2','flag4_3',
        'lw1_log' ,'lw2_log' ,'lw3_log','lw4_log','flagl1_2','flagl3_2','flagl4_3',
        'last_day_obj','first_day_obj']

def Get_Feature(log,user_info):
    result = pd.DataFrame()
    for i in xrange(user_info.shape[0]):
        if i % 10 == 0: print i
        # extract each enrollment_id infomation
        temp = log[log["enrollment_id"] == user_info["enrollment_id"][i]]
        
        #--------day related attributes
        #total day num
        day_num = len(set(temp["YMD"]))
        #start day
        start_day = min(temp["YMD"])
        #end day
        end_day = max(temp["YMD"])
        #log duration
        log_duration =  (end_day - start_day).days+1
        #attend ratio
        day_ratio = day_num/30
        #duration_ratio
        duration_ratio = log_duration/30
        
        #--------event related attributes
        #close_page num & avg close_page
        close_num = temp[temp["event"] == "page_close"].shape[0]
        avg_close = close_num/day_num
        avg_close2 = close_num/log_duration
        
        #navigate num & avg navigate
        nav_num = temp[temp["event"] == "navigate"].shape[0]
        avg_nav = nav_num/day_num
        avg_nav2 = nav_num/log_duration

        #access num & avg access
        acc_num = temp[temp["event"] == "access"].shape[0]
        avg_acc = acc_num/day_num
        avg_acc2 = acc_num/log_duration

        #wiki num & avg wiki
        wiki_num = temp[temp["event"] == "wiki"].shape[0]
        avg_wiki = wiki_num/day_num
        avg_wiki2 = wiki_num/log_duration

        #discuss num & avg discuss
        dis_num = temp[temp["event"] == "discussion"].shape[0]
        avg_dis = dis_num/day_num
        avg_dis2 = dis_num/log_duration

        #--------operation attributes
        operation_group = temp.groupby("YMD")
        operation_list = [len(group) for _,group in operation_group]
        operation_stats = pd.Series(operation_list).describe()
        
        operation_num = temp.shape[0]
        avg_operation = operation_stats["mean"]
        sd_opearation = operation_stats["std"]
        max_operation = operation_stats["max"]
        min_operation = operation_stats["min"]
        mid_operation = operation_stats["50%"]
        q25_operation = operation_stats["25%"]
        q75_operation = operation_stats["75%"]
        max_operation_ratio = max_operation/operation_num
        
        #--------object attributes
        object_list = [len(set(group["object"])) for _,group in operation_group]       
        object_stats = pd.Series(object_list).describe()
        
        uniq_object_num = len(set(temp["object"]))
        object_num = sum(object_list)
        avg_object = object_stats["mean"]
        sd_object = object_stats["std"]
        max_object = object_stats["max"]
        min_object = object_stats["min"]
        mid_object = object_stats["50%"]
        q25_object = object_stats["25%"]
        q75_object = object_stats["75%"]
        max_object_ratio = max_object/object_num
        
        #--------work problem attributes
        problem_data = temp[temp["event"]=="problem"]
        problem_group = problem_data.groupby("YMD")
        
        wproblem_list = [len(group) for _,group in problem_group]
        wproblem_stats = pd.Series(wproblem_list).describe()
        wproblem_num = problem_data.shape[0]
        avg_wproblem = wproblem_stats["mean"]
        sd_wproblem = wproblem_stats["std"]
        max_wproblem = wproblem_stats["max"]
        min_wproblem = wproblem_stats["min"]
        mid_wproblem = wproblem_stats["50%"]
        q25_wproblem = wproblem_stats["25%"]
        q75_wproblem = wproblem_stats["75%"]
        max_wproblem_ratio = max_wproblem/wproblem_num
        
        prob_day_ratio = wproblem_stats["count"]/day_num
        avg_wproblem2 = wproblem_num/day_num
        avg_wproblem3 = wproblem_num/log_duration
        
        #--------real problem attributes
        rproblem_list = [len(set(group["object"])) for _,group in problem_group]
        rproblem_stats = pd.Series(rproblem_list).describe()
        rproblem_num = len(set(problem_data["object"]))
        
        avg_rproblem = rproblem_stats["mean"]
        sd_rproblem = rproblem_stats["std"]
        max_rproblem = rproblem_stats["max"]
        min_rproblem = rproblem_stats["min"]
        mid_rproblem = rproblem_stats["50%"]
        q25_rproblem = rproblem_stats["25%"]
        q75_rproblem = rproblem_stats["75%"]
        max_rproblem_ratio = max_rproblem/rproblem_num
        
        avg_rproblem2 = rproblem_num/day_num
        avg_rproblem3 = rproblem_num/log_duration
        
        #--------work video attributes
        video_data = temp[temp["event"]=="video"]
        video_group = video_data.groupby("YMD")
        
        wvideo_list = [len(group) for _,group in video_group]
        wvideo_stats = pd.Series(wvideo_list).describe()
        wvideo_num = video_data.shape[0]
        avg_wvideo = wvideo_stats["mean"]
        sd_wvideo = wvideo_stats["std"]
        max_wvideo = wvideo_stats["max"]
        min_wvideo = wvideo_stats["min"]
        mid_wvideo = wvideo_stats["50%"]
        q25_wvideo = wvideo_stats["25%"]
        q75_wvideo = wvideo_stats["75%"]
        max_wvideo_ratio = max_wvideo/wvideo_num
        
        video_day_ratio = wvideo_stats["count"]/day_num
        avg_wvideo2 = wvideo_num/day_num
        avg_wvideo3 = wvideo_num/log_duration
        
        #--------real video attributes
        rvideo_list = [len(set(group["object"])) for _,group in video_group]
        rvideo_stats = pd.Series(rvideo_list).describe()
        rvideo_num = len(set(video_data["object"]))
        
        avg_rvideo = rvideo_stats["mean"]
        sd_rvideo = rvideo_stats["std"]
        max_rvideo = rvideo_stats["max"]
        min_rvideo = rvideo_stats["min"]
        mid_rvideo = rvideo_stats["50%"]
        q25_rvideo = rvideo_stats["25%"]
        q75_rvideo = rvideo_stats["75%"]
        max_rvideo_ratio = max_rvideo/rvideo_num
        
        avg_rvideo2 = rvideo_num/day_num
        avg_rvideo3 = rvideo_num/log_duration
        
        #--------log times realted attributes
        w1_log = temp[(temp["YMD"]>=start_day) & (temp["YMD"]<=(start_day + Week()))].shape[0]
        w2_log = temp[(temp["YMD"]>(start_day + Week())) & (temp["YMD"]<=(start_day + 2*Week()))].shape[0]
        w3_log = temp[(temp["YMD"]>(start_day + 2*Week())) & (temp["YMD"]<=(start_day + 3*Week()))].shape[0]
        w4_log = temp[(temp["YMD"]>(start_day + 3*Week())) & (temp["YMD"]<=(start_day + 4*Week()))].shape[0]
        
        flag1_2 = 1 if w2_log>w1_log else 0
        flag3_2 = 1 if w3_log>w2_log else 0
        flag4_3 = 1 if w4_log>w3_log else 0
        
        lw1_log = temp[(temp["YMD"]<=end_day) & (temp["YMD"]>=(end_day - Week()))].shape[0]
        lw2_log = temp[(temp["YMD"]<(end_day - Week())) & (temp["YMD"]>=(end_day - 2*Week()))].shape[0]
        lw3_log = temp[(temp["YMD"]<(end_day - 2*Week())) & (temp["YMD"]>=(end_day - 3*Week()))].shape[0]
        lw4_log = temp[(temp["YMD"]<(end_day - 3*Week())) & (temp["YMD"]>=(end_day - 4*Week()))].shape[0]
        
        flagl1_2 = 1 if lw1_log>lw2_log else 0
        flagl3_2 = 1 if lw2_log>lw3_log else 0
        flagl4_3 = 1 if lw3_log>lw4_log else 0
        
        last_day_obj = temp[temp["YMD"]==end_day].shape[0]
        first_day_obj = temp[temp["YMD"]==start_day].shape[0]     
        
        u_d = {'enrollment_id':user_info["enrollment_id"][i],'day_num':day_num, 'start_day':start_day, 'end_day':end_day,'log_duration' :log_duration,'day_ratio':day_ratio,'duration_ratio':duration_ratio,
        'close_num':close_num ,'avg_close':avg_close,'avg_close2':avg_close2, 'nav_num':nav_num,'avg_nav':avg_nav,'avg_nav2':avg_nav2,'acc_num':acc_num,'avg_acc':avg_acc, 'avg_acc2':avg_acc2,'wiki_num':wiki_num,'avg_wiki':avg_wiki,'avg_wiki2':avg_wiki2,'dis_num': dis_num,'avg_dis':avg_dis,'avg_dis2' :avg_dis2,
        'operation_num':operation_num,'avg_operation':avg_operation,'sd_opearation':sd_opearation,'max_operation':max_operation,'min_operation':min_operation,
        'mid_operation' : mid_operation,'q25_operation' :q25_operation,'q75_operation':q75_operation,'max_operation_ratio': max_operation_ratio,
       'uniq_object_num' : uniq_object_num,'object_num': object_num,'avg_object' : avg_object,'sd_object': sd_object,'max_object':max_object,
       'min_object' :min_object,'mid_object' :mid_object,'q25_object' : q25_object,'q75_object' :q75_object,'max_object_ratio' :max_object_ratio,
       'wproblem_num' : wproblem_num,'avg_wproblem':avg_wproblem,'sd_wproblem': sd_wproblem,'max_wproblem':max_wproblem,'min_wproblem' :min_wproblem,
       'mid_wproblem' :mid_wproblem,'q25_wproblem':q25_wproblem,'q75_wproblem' :q75_wproblem,'max_wproblem_ratio' :max_wproblem_ratio,'prob_day_ratio':prob_day_ratio,'avg_wproblem2': avg_wproblem2,'avg_wproblem3':avg_wproblem3,
       'avg_rproblem':avg_rproblem,'sd_rproblem':sd_rproblem,'max_rproblem':max_rproblem,'min_rproblem' :min_rproblem,
       'mid_rproblem':mid_rproblem,'q25_rproblem':q25_rproblem,'q75_rproblem': q75_rproblem,'max_rproblem_ratio':max_rproblem_ratio,'avg_rproblem2':avg_rproblem2,'avg_rproblem3':avg_rproblem3,
       'wvideo_num' : wvideo_num,'avg_wvideo':avg_wvideo,'sd_wvideo':sd_wvideo,'max_wvideo'  :max_wvideo,'min_wvideo' :min_wvideo,
       'mid_wvideo' :mid_wvideo,'q25_wvideo': q25_wvideo,'q75_wvideo':q75_wvideo,'max_wvideo_ratio' : max_wvideo_ratio,'video_day_ratio': video_day_ratio,'avg_wvideo2':avg_wvideo2,'avg_wvideo3':avg_wvideo3,     
       'rvideo_num' : rvideo_num, 'avg_rvideo' : avg_rvideo,'sd_rvideo' :sd_rvideo,'max_rvideo' : max_rvideo,'min_rvideo' : min_rvideo,'mid_rvideo' : mid_rvideo,'q25_rvideo':q25_rvideo,'q75_rvideo' : q75_rvideo,'max_rvideo_ratio':max_rvideo_ratio,'avg_rvideo2' : avg_rvideo2,'avg_rvideo3':avg_rvideo3,
       'w1_log'  : w1_log,'w2_log'  : w2_log,'w3_log'  :w3_log,'w4_log'  : w4_log,'flag1_2' : flag1_2,'flag3_2'  :flag3_2,'flag4_3'  :flag4_3,
       'lw1_log' : lw1_log,'lw2_log'  : lw2_log,'lw3_log'  :lw3_log,'lw4_log' : lw4_log,'flagl1_2' : flagl1_2,'flagl3_2'  : flagl3_2,'flagl4_3' : flagl4_3,'last_day_obj' : last_day_obj,'first_day_obj':first_day_obj    
       }
        # construct the current enrollment_id dataframe
        df = pd.DataFrame(u_d,index= [user_info["enrollment_id"][i]],columns=col)
        # equal to the rbind funtion in R
        result = pd.concat([result,df])
    return result


begin = datetime.now()

ans1 = Get_Feature(log_train_df,enrollment_train_df)
ans1.to_csv("train.csv",index = False)


ans1 = Get_Feature(log_test_df,enrollment_test_df)
ans1.to_csv("test.csv",index = False)

print datetime.now()-begin