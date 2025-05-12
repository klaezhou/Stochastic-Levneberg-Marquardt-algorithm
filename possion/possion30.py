"""PINN_LM"""
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
# %matplotlib widget
from matplotlib import pyplot as plt
from numpy import matrix as mat
from torch.nn.modules import loss
from functorch import make_functional, vmap, grad, jacrev, hessian
import jax
import time
from matplotlib.ticker import FuncFormatter
pi=np.pi
class PINN_LM:
    """
    同为2 维 ： 需要修改的方法：
    newspampling()
    LM(): fx_fun() 
          J_func()
    
          plot_l2error.py 中exact solution
    """
    def __init__(self):
        self.input_size = 2
        self.hidden_size =30
        self.output_size = 1
        self.depth = 1
        global device
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # 选择使用GPU还是CPU
        self.model = Network(self.input_size, self.hidden_size,  self.output_size, self.depth, act=  torch.nn.Tanh() ).double().to(device)  # 定义神经网络
        self.p_number = self.input_size * self.hidden_size + self.hidden_size + (self.hidden_size * self.hidden_size + self.hidden_size)*2 +self.hidden_size * self.output_size  # 参数的个数
        print('model # of parameters',self.p_number)
        
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
            random_samples = 15
            torch.manual_seed(1+i)
            x = (upper_boundx - lower_boundx) * torch.rand(random_samples) + lower_boundx 
            torch.manual_seed(2+i)
            y = (upper_boundy - lower_boundy) * torch.rand(random_samples) + lower_boundy  
            self.X_inside = torch.stack(torch.meshgrid(x, y)).reshape(2, -1).T  
            self.X_inside=self.X_inside.to(device)
            self.X_inside.requires_grad = True
            self.X_inside_num=self.X_inside.size(0) 
            
            random_samples = 20
            torch.manual_seed(4+i)
            y = (upper_boundy - lower_boundy) * torch.rand(random_samples) + lower_boundy 
            bc1 = torch.stack(torch.meshgrid(torch.tensor(lower_boundx).double(), y)).reshape(2, -1).T  # x=-1边界
            bc2 = torch.stack(torch.meshgrid(torch.tensor(upper_boundx).double(), y)).reshape(2, -1).T  # x=+1边界
            random_samples=20
            torch.manual_seed(3+i)
            x = (upper_boundx - lower_boundx) * torch.rand(random_samples) + lower_boundx  
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

    
    def LM(self,opt_num, step,deterministic=True,mu=1):
        """ 
        Parameters:
        opt_num -  优化的参数数量 
        step: LM 下降的次数 
        deterministic: 是否要严格下降 True 为是
        mu:  damp parameter >0 为LM method， 0 为Gauss-Newton => deterministic=False
        需要修改的部分：fx_fun 函数; J_func 函数; 
        """

        if mu==0:
            deterministic=False
        p_number = self.p_number
        params = torch.cat([p.view(-1) for p in self.model.parameters()], dim=0).to(device) # 把模型参数展平成一维向量

        
    
        def f(params, input_data):
            
            with torch.no_grad():
                a = self.hidden_size * self.input_size
                self.model.layer1.weight.data = params[:a].reshape(self.hidden_size, self.input_size).clone()  # layer1的权重
                self.model.layer1.bias.data = params[a:a + self.hidden_size].clone()  # layer1的偏置
                a += self.hidden_size
                self.model.layer2.weight.data = params[a:a + self.hidden_size * self.hidden_size].reshape(self.hidden_size, self.hidden_size).clone()  # layer2的权重
                a += self.hidden_size * self.hidden_size
                self.model.layer2.bias.data = params[a:a + self.hidden_size].clone()  # layer2的偏置
                a += self.hidden_size
                self.model.layer3.weight.data = params[a:a + self.hidden_size * self.hidden_size].reshape(self.hidden_size, self.hidden_size).clone()  # layer2的权重
                a += self.hidden_size * self.hidden_size
                self.model.layer3.bias.data = params[a:a + self.hidden_size].clone()  # layer2的偏置
                a += self.hidden_size
                self.model.layer4.weight.data = params[a:a + self.output_size * self.hidden_size].reshape(self.output_size, self.hidden_size).clone()  # layer3的权重
                a += self.output_size * self.hidden_size
                # self.model.layer3.bias.data = params[a].clone()  # layer3的偏置
            
            model_output=torch.vmap(self.model)(input_data)    
            
            return model_output
        
        def fx_fun(params)->np.array:
            
            f_inside = f(params, self.X_inside)  # 更换模型中的参数
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
            
        
            fx = du_dyy+du_dxx + 904* torch.sin(30*self.X_inside[:,0])*torch.sin(2*self.X_inside[:,1])
            fx=fx.to(device)
            fx=fx.view(-1)
            
            f_bd = f(params, self.X_boundary)
            
            fx_bd = f_bd.view(-1)
            

            fx = torch.cat((fx, fx_bd), dim=0)
            fx=fx.t()
            return fx

        def F_fun(fx)->torch.tensor:
            '''计算L2范数'''
            F_p = 0
            for i in range(self.X_inside_num+self.X_boundary_num):
                F_p += (fx[i]) ** 2 
            F_p /= (self.X_inside_num+self.X_boundary_num)
            return F_p
        

        def J_func(params)->torch.tensor:
            J = torch.zeros(self.X_inside_num+self.X_boundary_num, p_number).to(device)# 初始化雅可比矩阵
            params.requires_grad_(True)
            def Inter(params, input):
                
                f_inside = f(params, input)  # 更换模型中的参数
                
                func_model, func_params = make_functional(self.model)
                def fm(x, func_params):
                    fx = func_model(func_params, x)
                    return fx.squeeze(0).squeeze(0)
                def floss(func_params,input):
                    f_inside=fm(input, func_params) 
                    d1u = jacrev(fm)(input,func_params)         
                    d2u = jacrev(jacrev(fm))(input,func_params)
                    
                      
                    du_dyy=d2u[1][1] 
                    du_dxx=d2u[0][0]     
                    
                    fx = du_dyy+du_dxx + 904*torch.sin(30*input[0])*torch.sin(2*input[1])
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
                f_bound=f(params,input)
                
                func_model, func_params = make_functional(self.model)
                def fm(x, func_params):
                    fx = func_model(func_params, x)
                    return fx.squeeze(0).squeeze(0)
                def floss(func_params,input):
                    fx = fm(input, func_params)

                    return fx
                
                per_sample_grads =vmap(jacrev(floss), (None, 0))(func_params, input)
                
                cnt=0
                for g in per_sample_grads: 
                    g = g.detach()
                    J_d = g.reshape(len(g),-1) if cnt == 0 else torch.hstack([J_d,g.reshape(len(g),-1)])
                    cnt = 1
                
                result = J_d.detach()
                
                return result
            
            
            J[range(self.X_inside_num), :] = Inter(params, self.X_inside)
            J[range(self.X_inside_num, self.X_inside_num + self.X_boundary_num), :] = Bound(params, self.X_boundary)
            
            return J
        
        
        #参数准备
        lmin=1e-9
        lmax=1e12
        k = 0
        kmax = step
        p = params.to(device)
        J = J_func(p)
        A = torch.matmul(J.t(),J)
        F_p = torch.tensor(10).to(device)
        F_pnew = 0
        alpha = 1
        lambda_up = 5
        lambda_down = 0.2
        yi=1e-7
        yi2=1e-15
        diag = torch.eye(p_number)
        fx = fx_fun(p)
        mu=torch.norm(fx, p=2).cpu().detach().item()**2
        elapsed_time_ms=0
        ####随机选择部分参数进行优化
        selected_columns = np.random.choice(p_number, opt_num, replace=False)
        ####
        if deterministic:
            print('begin a new iteration')
            '''严格lm'''
            while (k < kmax):  # (not found)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                k = k + 1
                
                J_opt=J[:,selected_columns]
                
                A_opt= torch.matmul(J_opt.t(),J_opt)
                diag=torch.eye(A_opt.shape[0]).to(device)
                H = A_opt + mu * diag
                
                fx = fx_fun(p)
                gkF=torch.matmul(J_opt.t(),fx)
                # gk=torch.matmul(J.t(),fx)
 
                
                # LU, pivots = torch.linalg.lu_factor(H)
                # # 生成单位矩阵，确保是二维矩阵
                # LU=LU.to(device)
                # pivots = pivots.to(device)
                # h_lm = torch.linalg.lu_solve(LU, pivots, -gkF.unsqueeze(1))
                try:              
                    h_lm = torch.linalg.solve(H, -gkF)   
                    #LU decomposition
                    #pinv 
                    # h_lm = torch.linalg.pinv(H) @ -gkF
                   
                    

                except:
                    print('singular matrix')
                    print("mu:", mu)
                    
                    break
                    print(-gkF)
                
                if (torch.abs(F_p - F_pnew)/torch.tensor(F_p).to(device)< 1e-15):  # 满足收敛条件
                    print('converge in para updates')
                    break
                else:
                    p_new=p.clone()
                    p_new[selected_columns]+= alpha * torch.squeeze(h_lm)
                                        
                    F_p = F_fun(fx)                    
                    fx_new = fx_fun(p_new)
                    F_pnew = F_fun(fx_new)             
                    o = F_p - F_pnew
                    o_=torch.matmul(gkF.t(),h_lm)+1/2*torch.matmul(h_lm.t(),torch.matmul(A_opt,h_lm))+1/2*mu*torch.norm(h_lm, p=2)
                    
                    if o>0 and o/o_ > yi and torch.norm(gkF,p=2)**2>yi2/mu:
                        #print('Zhuyi:',torch.sqrt(1-1/(2*torch.norm(gk,p=2)**4 * mu**2))) #torch.sqrt(1-1/(2*torch.norm(gk,p=2)**4 * mu**2))
                        self.loss_record[self.loss_iter] = float(F_pnew.item())
                        self.loss_iter += 1
                        if k%1==0:
                            print("steps ", k, end=' ')
                            print('accept move')
                            print("Loss =", F_p.item(), end=' ')
                            print("Loss new=", F_pnew.item())
                            print(f'Elapsed: {elapsed_time_ms:.1f}ms')  
                        p = p_new
                        J = J_func(p)  # update J
                        mu = max(mu * lambda_down,lmin)  # lower limit u =1e-11
                       
                        
                    else:
                        if k % 10 == 0:print("reject move")
                        mu = min(mu * lambda_up,lmax)  # Up limit u =1e11
                    end_event.record()
                    torch.cuda.synchronize()  # Wait for the events to be recorded!
                    elapsed_time_ms = start_event.elapsed_time(end_event)
                    self.time_record[self.time_iter] = elapsed_time_ms
                    self.time_iter += 1
                    
                
        else:
            pass
                    
        self.avg_time = np.sum(self.time_record[self.time_record != 0])        
                
        
    
    
    def plt(self):
        pass

    def error(self):
        pass
    







class Network(nn.Module):
        def __init__(
            self,
            input_size, # 输入层神经元数
            hidden_size, # 隐藏层神经元数
            output_size, # 输出层神经元数
            depth, # 隐藏层数
            act=torch.nn.Tanh(), # 输入层和隐藏层的激活函数
        ):
            super(Network, self).__init__()

            # 输入层
            self.layer1 = nn.Linear(in_features=input_size, out_features=hidden_size)
            #self.relu = nn.ReLU() torch.nn.Tanh()
            self.tanh = act
            # 隐藏层
            self.layer2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
            self.layer3 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
            #self.relu = nn.ReLU()
            self.tanh = act
            # 输出层
            self.layer4 = nn.Linear(in_features=hidden_size, out_features=output_size,bias=False)



        def forward(self, x):
            x = self.layer1(x)
            #x = self.relu(x)
            x= self.tanh(x)
            x = self.layer2(x)
            #x = self.relu(x)
            x = self.tanh(x)
            x = self.layer3(x)
            x = self.tanh(x)
            x = self.layer4(x)

            return x
        