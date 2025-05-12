
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
# %matplotlib widget
from matplotlib import pyplot as plt
from numpy import matrix as mat
from torch.nn.modules import loss
from torch import  vmap
from torch.func import jacrev ,functional_call
from functorch import make_functional, vmap, grad, jacrev, hessian
import time
from matplotlib.ticker import FuncFormatter
import collections

pi=np.pi
class PINN_Toy:
    """
    同为2 维 ： 需要修改的方法：
    newspampling()
    LM(): fx_fun() 
          J_func()
          plot_l2error.py 中exact solution
    """
    def __init__(self,cuda_num=0):
        self.input_size = 2
        self.hidden_size1=100
        self.hidden_size2=100
        self.output_size = 1
        self.depth = 1
        global device
        device = torch.device(f"cuda:{cuda_num}") if torch.cuda.is_available() else torch.device("cpu")  # 选择使用GPU还是CPU
        self.model = Network(self.input_size, self.hidden_size1, self.hidden_size2,    self.output_size, self.depth, act=  torch.nn.Tanh() ).double().to(device)  # 定义神经网络
        self.p_number = self.input_size * self.hidden_size1 + self.hidden_size1 + (self.hidden_size1 * self.hidden_size2 + self.hidden_size2) + (self.hidden_size2 * self.hidden_size2 + self.hidden_size2)  +self.hidden_size2 * self.output_size  # 参数的个数
        print('model # of parameters',self.p_number)
        torch.set_default_dtype(torch.float64)
        ## Initial weights
        def initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()
                        # m.bias.data.normal_(mean=0.0, std=1.0)

        initialize_weights(self.model)


        
        
        self.loss_record=np.zeros(100000)
        self.loss_iter=0
        self.Sampledata=np.array([])
        self.time_iter=0
        self.time_record=np.zeros(100000)
        
    def new_sampling(self,i):
            """
            生成新的采样点
            i - random seed 
            points_num - # of points inside 
            """
            # 指定区间
            torch.set_default_dtype(torch.float64)
            lower_boundx = 0
            upper_boundx = pi
            lower_boundy=0
            upper_boundy=pi
            random_samples = 40
            torch.manual_seed(1+i)
            # x = (upper_boundx - lower_boundx) * torch.rand(random_samples) + lower_boundx 
            y = torch.linspace(lower_boundy, upper_boundy, random_samples)
            torch.manual_seed(2+i)
            # y = (upper_boundy - lower_boundy) * torch.rand(random_samples) + lower_boundy  
            x = torch.linspace(lower_boundx, upper_boundx, random_samples)
            self.X_inside = torch.stack(torch.meshgrid(x, y)).reshape(2, -1).T  
            self.X_inside=self.X_inside.to(device)
            self.X_inside.requires_grad = True
            self.X_inside_num=self.X_inside.size(0) 
            
            random_samples = 100
            torch.manual_seed(4+i)
            # y = (upper_boundy - lower_boundy) * torch.rand(random_samples) + lower_boundy 
            y = torch.linspace(lower_boundy, upper_boundy, random_samples)
            bc1 = torch.stack(torch.meshgrid(torch.tensor(lower_boundx).double(), y)).reshape(2, -1).T  # x=-1边界
            bc2 = torch.stack(torch.meshgrid(torch.tensor(upper_boundx).double(), y)).reshape(2, -1).T  # x=+1边界
            random_samples=40
            torch.manual_seed(3+i)
            # x = (upper_boundx - lower_boundx) * torch.rand(random_samples) + lower_boundx  
            x = torch.linspace(lower_boundx, upper_boundx, random_samples)
            bc3 = torch.stack(torch.meshgrid(x, torch.tensor(lower_boundy).double())).reshape(2, -1).T # y=0边界
            bc4 = torch.stack(torch.meshgrid(x, torch.tensor(upper_boundy).double())).reshape(2, -1).T # y=0边界

            self.X_boundary = torch.cat([bc1, bc2, bc3,bc4])  # 将所有边界处的时空坐标点整合为一个张量
            
            self.X_boundary_num=self.X_boundary.size(0)
            
            self.X_inside_num=self.X_inside.size(0)

            self.X_boundary=self.X_boundary.to(device)

            self.X_inside=self.X_inside.to(device)

            self.X_boundary.requires_grad = True
            
            self.X_boundary=self.X_boundary.double()
            self.X_inside=self.X_inside.double()
            print("model # of sampling points:",self.X_inside_num+self.X_boundary_num)

    def f(self,params, input_data):
            
                with torch.no_grad():
                    a = self.hidden_size1 * self.input_size
                    self.model.model[0].weight.data = params[:a].reshape(self.hidden_size1, self.input_size).clone()  # layer1的权重
                    self.model.model[0].bias.data = params[a:a + self.hidden_size1].clone()  # layer1的偏置
                    a += self.hidden_size1
                    self.model.model[2].weight.data = params[a:a + self.hidden_size1 * self.hidden_size2].reshape(self.hidden_size2, self.hidden_size1).clone()  # layer2的权重
                    a += self.hidden_size1 * self.hidden_size2
                    self.model.model[2].bias.data = params[a:a + self.hidden_size2].clone()  # layer2的偏置
                    a += self.hidden_size2
                    self.model.model[4].weight.data = params[a:a + self.hidden_size2 * self.hidden_size2].reshape(self.hidden_size2, self.hidden_size2).clone()  # layer2的权重
                    a += self.hidden_size2 * self.hidden_size2
                    self.model.model[4].bias.data = params[a:a + self.hidden_size2].clone()  # layer2的偏置
                    a += self.hidden_size2
                    self.model.model[6].weight.data = params[a:a + self.output_size * self.hidden_size2].reshape(self.output_size, self.hidden_size2).clone()  # layer3的权重
                    a += self.output_size * self.hidden_size2
                
                model_output=torch.vmap(self.model)(input_data)    
            
                return model_output
        
    def fx_fun(self,params)->np.array:
            
            f_inside = self.f(params, self.X_inside)  # 更换模型中的参数
            f_inside.require_grad = True
            du_dX = torch.autograd.grad(
                inputs=self.X_inside,
                outputs=f_inside,
                grad_outputs=torch.ones_like(f_inside),
                retain_graph=True,
                create_graph=True
                )[0][:,0]  
            du_dY = torch.autograd.grad(
                inputs=self.X_inside,
                outputs=f_inside,
                grad_outputs=torch.ones_like(f_inside),
                retain_graph=True,
                create_graph=True
            )[0][:,1]  
            du_dxx = torch.autograd.grad(
                inputs=self.X_inside,
                outputs=du_dX,
                grad_outputs=torch.ones_like(du_dX),
                retain_graph=True,
                create_graph=True
            )[0][:, 0]
            du_dyy = torch.autograd.grad(
                inputs=self.X_inside,
                outputs=du_dY,
                grad_outputs=torch.ones_like(du_dY),
                retain_graph=True,
                create_graph=True
            )[0][:, 1]
            
        
            fx = du_dyy+du_dxx + 68* torch.sin(8*self.X_inside[:,0])*torch.sin(2*self.X_inside[:,1])
            fx=fx.to(device)
            fx=fx.view(-1)
            
            f_bd = self.f(params, self.X_boundary)
            
            fx_bd = f_bd.view(-1)
            

            fx = torch.cat((fx, fx_bd), dim=0)
            fx=fx.t()
            return fx

    def J_func(self,params,p_number)->torch.tensor:
            J = torch.zeros(self.X_inside_num+self.X_boundary_num, p_number).to(device)# 初始化雅可比矩阵
            
            def Inter(params, input):
                
                with torch.no_grad():
                    f_inside = self.f(params, input)  # 更换模型中的参数
                
                func_model, func_params = make_functional(self.model)
                
                def fm(x, func_params):
                    fx = func_model(func_params, x)
                    return fx.squeeze(0).squeeze(0)
                def floss(func_params,input):      
                    d2u = jacrev(jacrev(fm))(input,func_params)
                    
                      
                    du_dyy=d2u[1][1] 
                    du_dxx=d2u[0][0]     
                    
                    fx = du_dyy+du_dxx + 68*torch.sin(8*input[0])*torch.sin(2*input[1])
                    # print('du_dY',du_dY)
                    return fx
                    
                
                per_sample_grads =vmap(jacrev(floss), (None, 0))(func_params, input)
                
            
                cnt=0
                
                for g in per_sample_grads: 
                    g = g.detach()
                    J_d = g.reshape(len(g),-1) if cnt == 0 else torch.hstack([J_d,g.reshape(len(g),-1)])
                    cnt = 1
                
                result= J_d.detach()
                

                return result
                
            
            def Bound(params,input):
                with torch.no_grad():
                    f_bound=self.f(params,input)
                
                func_params=self.model.state_dict()
                
                
                
        
                def fm(params,input):
                    return functional_call(self.model,params,input.unsqueeze(0)).squeeze(0)
                
                def floss(func_params,input):
                    fx = fm(func_params, input)
                    return fx
                
                
                
                
                per_sample_grads =vmap(jacrev(floss,0), (None, 0))(func_params, input)
                
                cnt=0
                for k,g in per_sample_grads.items(): 
                    
                    g = g.detach()
                    J_d = g.reshape(len(g),-1) if cnt == 0 else torch.hstack([J_d,g.reshape(len(g),-1)])
                    cnt = 1
                
                result = J_d.detach()
                
                return result
            
            
            J[range(self.X_inside_num), :] = Inter(params, self.X_inside)
            J[range(self.X_inside_num, self.X_inside_num + self.X_boundary_num), :] = Bound(params, self.X_boundary)
            return J
        
    @staticmethod    
    def grad(f,param):
            return torch.autograd.grad(f,param)    
        
        
                                  
                    
                    
                    
                
       
                    
    


class Network(nn.Module):
        def __init__(
            self,
            input_size, # 输入层神经元数
            hidden_size1, # 隐藏层神经元数
            hidden_size2, # 隐藏层神经元数
            output_size, # 输出层神经元数
            depth, # 隐藏层数
            act=torch.nn.Tanh(), # 输入层和隐藏层的激活函数
        ):
            super(Network, self).__init__()

        
            self.model = nn.Sequential(
                collections.OrderedDict
                    (
                    [("layer1",nn.Linear(in_features=input_size, out_features=hidden_size1)),
                ("activation1",act),
                ("layer2",nn.Linear(in_features=hidden_size1, out_features=hidden_size2)),
                ("activation2",act),
                ("layer3",nn.Linear(in_features=hidden_size2, out_features=hidden_size2)),
                ("activation3",act),
                ("layer4",nn.Linear(in_features=hidden_size2, out_features=output_size, bias=False))]
                )
                
            )

        def forward(self, x):
            x = self.model(x)
            return x

        