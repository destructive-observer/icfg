# from visdom import Visdom
import visdom
'''
单条追踪曲线设置
'''
# print(dir(visdom))
viz = visdom.Visdom(env='model_1')  # 初始化visdom类
viz.line([0.],    ## Y的第一个点坐标
        [0.],    ## X的第一个点坐标
        win="train loss",    ##窗口名称
        opts=dict(title='train_loss')  ## 图像标例
        )  #设置起始点
'''
模型数据
'''
viz.line([1.],   ## Y的下一个点坐标
        [1.],   ## X的下一个点坐标
        win="train loss", ## 窗口名称 与上个窗口同名表示显示在同一个表格里
        update='append'   ## 添加到上一个点后面
        )  
        
# Setting up a new session...

# 'train loss'