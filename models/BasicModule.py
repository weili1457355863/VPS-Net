"""
  author: lw
  email: hnu-lw@foxmail.com
  data: 19-4-14 下午8:23 
  description:封装nn.Module,主要提供save和load两种方法
"""
import torch
import torch.nn as nn
import time
class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))

    def load(self,path):
        '''
        :param path:加载模型的地址
        :return:
        '''
        self.load_state_dict(torch.load(path))

    def save(self,model_name, epoch):
        '''
        :param name:模型名字，default：’name+time‘
        :return:
        '''

        prefix='./checkpoints/'+model_name+'_'+str(epoch)+'_'
        name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(),name)
        return name