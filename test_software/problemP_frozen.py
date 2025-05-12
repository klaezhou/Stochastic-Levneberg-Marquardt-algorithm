"""PINN_LM"""
import torch
import torch.nn as nn
import collections 
import numpy as np
# %matplotlib widget
from matplotlib import pyplot as plt
from numpy import matrix as mat
from torch.nn.modules import loss

from torch import  vmap
from torch.func import jacrev ,functional_call,vjp
from functorch import make_functional, vmap, grad, jacrev, hessian
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
    def __init__(self,cuda_num=5):
        self.input_size = 2
        self.hidden_size1=40
        self.hidden_size2=40
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

    
    def LM(self,opt_num, step,modify=True):
        """ 
        Parameters:
            opt_num -- int ; 优化的参数数量 
            step -- int ; LM 下降的次数 
            modify -- bool ; 是否采用modify lm
        需要修改的部分:fx_fun 函数; J_func 函数; 
        """
        
        p_number = self.p_number
        params = torch.cat([p.view(-1) for p in self.model.parameters()], dim=0).to(device) # 把模型参数展平成一维向量

        
    
        def f(params, input_data):
            
                with torch.no_grad():
                    a = self.hidden_size1 * self.input_size
                    self.model.model.layer1.weight.data = params[:a].reshape(self.hidden_size1, self.input_size).clone()  # layer1的权重
                    self.model.model.layer1.bias.data = params[a:a + self.hidden_size1].clone()  # layer1的偏置
                    a += self.hidden_size1
                    self.model.model.layer2.weight.data = params[a:a + self.hidden_size1 * self.hidden_size2].reshape(self.hidden_size2, self.hidden_size1).clone()  # layer2的权重
                    a += self.hidden_size1 * self.hidden_size2
                    self.model.model.layer2.bias.data = params[a:a + self.hidden_size2].clone()  # layer2的偏置
                    a += self.hidden_size2
                    self.model.model.layer3.weight.data = params[a:a + self.hidden_size2 * self.hidden_size2].reshape(self.hidden_size2, self.hidden_size2).clone()  # layer2的权重
                    a += self.hidden_size2 * self.hidden_size2
                    self.model.model.layer3.bias.data = params[a:a + self.hidden_size2].clone()  # layer2的偏置
                    a += self.hidden_size2
                    self.model.model.layer4.weight.data = params[a:a + self.output_size * self.hidden_size2].reshape(self.output_size, self.hidden_size2).clone()  # layer3的权重
                    a += self.output_size * self.hidden_size2
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
            
        
            fx = du_dyy+du_dxx + 68* torch.sin(8*self.X_inside[:,0])*torch.sin(2*self.X_inside[:,1])
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
                with torch.no_grad():
                    f_inside = f(params, input)  # 更换模型中的参数
                
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
        def J_func_modify(params,layer)->torch.tensor:
            """
            Jacobi on specific layer
            
            Parameters :
               params -- params
               layer -- int;  Get the the derivative of the 'layer's layer parameters.
            Returns:
               Jocobian matirx with parameters in laye
            
            """
            
            params.requires_grad_(True)
            
            func_params=dict(self.model.model.named_parameters())
            list_params=list(func_params.values())
            def fm(input,params):
                    return functional_call(self.model.model,params,input).squeeze(0)
            
            
            
            grad_parameter= list_params[layer-1]
            grad_dim=len(grad_parameter.view(-1))
            
            J = torch.zeros(self.X_inside_num+self.X_boundary_num, grad_dim).to(device)# 初始化雅可比矩阵
            def Inter(params, input):
                # def fmiver(input,params):
                #         return fm(params,input) 
                with torch.no_grad():
                    f_inside = f(params, self.X_boundary)  # 更换模型中的参数
                def floss(func_params,input):      
                    
                    # d2u = jacrev(jacrev(fm))(input,func_params)[0]
                    # print(d2u.shape)
                    # du_dyy=d2u[1][1] 
                    # du_dxx=d2u[0][0]     
                    # print(input.shape)
                    output=fm(input,func_params)
                    # print(output.shape)
                    du_d1 = torch.autograd.grad(
                        inputs=input,
                        outputs=output,
                        grad_outputs=torch.ones((self.X_inside_num,1)).to(device),
                        retain_graph=True,
                        create_graph=True
                        )[0]
                    
                    du_d1.unsqueeze(-1)
                    # print(du_d1[:,[0]].shape)
                    du_dxx = torch.autograd.grad(
                        inputs=input,
                        outputs=du_d1[:,[0]],
                        grad_outputs=torch.ones((self.X_inside_num,1)).to(device),
                        retain_graph=True,
                        create_graph=True
                        )[0]
                    du_dyy = torch.autograd.grad(
                        inputs=input,
                        outputs=du_d1[:,[1]],
                        grad_outputs=torch.ones((self.X_inside_num,1)).to(device),
                        retain_graph=True,
                        create_graph=True
                        )[0]
                    
                    
                    fx = du_dyy+du_dxx + 68*torch.sin(8*input[0])*torch.sin(2*input[1])
                    # print(fx.shape)
                    # torch.reshape(fx)
                    
                    return fx
                
                # loss=floss(func_params,input)
                # print("loss,",loss.shape)
                basis_vector=torch.eye(self.X_inside_num).to(device)
                basis_vector=basis_vector
                # basis_vector=torch.randn(self.X_inside_num).to(device)
                def grad(inputs,v):
                    # print('i',inputs.shape)
                    loss=floss(func_params,input)
                    # print("s",loss.shape)
                    # return torch.func.vjp(loss)
                    # return jacrev(,0)(func_params,inputs)
                    return torch.autograd.grad(loss,grad_parameter,torch.ones_like(loss),create_graph=True,retain_graph=True,allow_unused=True) 
                
                
                gradient=vmap(grad,in_dims=(None,0))(input,basis_vector)
                
                # print('g',gradient)
                cnt=0
                
                for g in gradient: 
                    g = g.detach()
                    J_d = g.reshape(len(g),-1) if cnt == 0 else torch.hstack([J_d,g.reshape(len(g),-1)])
                    cnt = 1
                
                result= J_d.detach()
                
                
                return result
            
            def Bound(params,input):
                
                
                def floss(func_params,input):
                    fx = fm(input, func_params)

                    return fx
                loss=floss(func_params,input)    
                basis_vector=torch.eye(self.X_boundary_num).to(device)
                basis_vector=basis_vector.unsqueeze(-1)
                def grad(v):
                    return torch.autograd.grad(loss,grad_parameter,v,retain_graph=True)
                
                gradient=vmap(grad)(basis_vector)
            
                cnt=0
                
                for g in gradient: 
                    g = g.detach()
                    J_d = g.reshape(len(g),-1) if cnt == 0 else torch.hstack([J_d,g.reshape(len(g),-1)])
                    cnt = 1
                
                result= J_d.detach()
                
                
                return result
            
            
            J[range(self.X_inside_num), :] = Inter(params, self.X_inside)
            J[range(self.X_inside_num, self.X_inside_num + self.X_boundary_num), :] = Bound(params, self.X_boundary)
            
            return J
        
        def get_weight_indices(n, model):
            # 将模型所有参数展开成一个向量，逐层累积参数量
            if n%2==0:
                layer=n/2
            else:
                layer=n//2+1
            start_index = 0
            
            for name, param in model.named_parameters():
                
                # 只对权重参数进行处理（忽略 bias）
                if f"layer{layer}.weight" in name and n%2==1:
                    end_index = start_index + param.numel() 
                    return range(start_index ,end_index)
                elif f"layer{layer}.bias" in name and n%2==0:
                    end_index = start_index + param.numel() 
                    return range(start_index ,end_index)
                # 更新索引范围
                start_index += param.numel()
            # 如果找不到对应的层，抛出异常
            raise ValueError(f"Layer {n} does not exist or does not have a weight parameter.")
        random_number=3
        selected_columns=get_weight_indices(random_number,self.model.model)
        print(selected_columns)
        #参数准备
        lmin, lmax, k, kmax, p, F_pnew, alpha, lambda_up, lambda_down, yi, yi2 = 1e-15, 1e15, 0, step, params.to(device), 1, 1, 2, 0.5, 1e-15, 1e-15
        mu=torch.tensor(100).to(device)
        print("pp",p[selected_columns].shape)
        
        ##
        J=J_func_modify(p,3)
        print('JJ',J.shape)
        J = J_func(p)
        ##
        J = J_func(p)
        J_opt=J
        A = torch.matmul(J.t(),J)
        F_p = torch.tensor(10).to(device)
        diag = torch.eye(p_number)
        fx = fx_fun(p)
        
        elapsed_time_ms=0
        ####随机选择部分参数进行优化
        zone=False
     
        ####
        start_event = torch.cuda.Event(enable_timing=True)
        step_event=torch.cuda.Event(enable_timing=True)
        endstep=torch.cuda.Event(enable_timing=True) #
        start_event.record()
        if modify:
            print('begin a new iteration')
            '''严格lm'''
            while (k < kmax):        
                end_event = torch.cuda.Event(enable_timing=True)
                k = k + 1
                
                A_opt= torch.matmul(J_opt.t(),J_opt)
                diag=torch.eye(A_opt.shape[0]).to(device)
                # print(torch.max(A_opt))
                # print('mu',mu)
                H = A_opt + mu * diag
                
                fx = fx_fun(p)
                
                with torch.no_grad():
                    gkF=torch.matmul(J_opt.t(),fx)
                    gk=gkF
                
                try:              
                    h_lm = torch.linalg.solve(H, -gkF)                
                except:
                    print('singular matrix')   
                if  F_pnew<1e-9:  # 满足收敛条件
                    print('###########################################converge iteration:',k)
                    print('converge in para updates')
                    break
                else:
                    p_new=p.clone()  
                    if zone:
                        p_new[selected_columns]+= alpha * torch.squeeze(h_lm)
                    else:
                        p_new+=alpha*torch.squeeze(h_lm)
                                        
                    F_p = F_fun(fx)   
                                  
                    fx_new = fx_fun(p_new)
                      
                    F_pnew = F_fun(fx_new)             
                    o = F_p - F_pnew
                     
                    o_=torch.matmul(gkF.t(),h_lm)+1/2*torch.matmul(h_lm.t(),torch.matmul(A_opt,h_lm))+1/2*mu*torch.norm(h_lm, p=2)
                    ratio=o/o_
                   
                    if ratio > yi and torch.norm(gkF,p=2)**2>yi2/mu:
                        self.loss_record[self.loss_iter] = float(F_pnew.item())
                        self.loss_iter += 1
                        self.time_record[self.time_iter] = elapsed_time_ms/1000
                        self.time_iter += 1
                        
                        if k%1==0:
                            print("steps ", k, end=' ')
                            print('accept move')
                            print("Loss =", F_p.item(), end=' ')
                            print("Loss new=", F_pnew.item())
                            print(f'Elapsed: {elapsed_time_ms:.1f}ms')  
                        p = p_new
                        
                        mu = max(mu * lambda_down,lmin)  
                        evaluation=(1/(2*torch.norm(gk,p=2)**4 * mu**2))
                        print('lamda',mu)
                        if k%1==0:
                            print('evaluation:',evaluation.item())
                            if evaluation.item() >=1e-15  and not zone :
                                # update J
                                J = J_func(p)
                                J_opt=J


                    
                            else:
                                print('modify step')
                                
                                
                                if k%100==0:
                                    random_number1= torch.randint(1, 7, (1,)).item()
                                    
                                    mu=0.1
                                    selected_columns=get_weight_indices(random_number,self.model.model)
                                    print("len",len(selected_columns))
                                
                                J_opt=J_func_modify(p,random_number)
                                # J = J_func(p)
                                # J_opt=J
                                if not zone:
                                    zone=True
                                    mu=10
                                
                                
                                     
                        
                    else:
                        if k % 1 == 0:
                            print("reject move")
                            print("Loss =", F_p.item(), end=' ')
                        mu = min(mu * lambda_up,lmax)  
                    end_event.record()
                    torch.cuda.synchronize()  # Wait for the events to be recorded!
                    elapsed_time_ms = start_event.elapsed_time(end_event)
                    
                    
                
        else:
            print('begin a new iteration')
            '''严格lm'''
            while (k < kmax):  # (not found)
                end_event = torch.cuda.Event(enable_timing=True)
                k = k + 1
                
                J_opt=J[:,selected_columns]
                
                A_opt= torch.matmul(J_opt.t(),J_opt)
                diag=torch.eye(A_opt.shape[0]).to(device)
                H = A_opt + mu * diag
                
                fx = fx_fun(p)
                with torch.no_grad():
                    gkF=torch.matmul(J_opt.t(),fx)
                    gk=torch.matmul(J.t(),fx)
                try:              
                    h_lm = torch.linalg.solve(H, -gkF)             
                except:
                    print('singular matrix')
                
                if  F_pnew<1e-9:  # 满足收敛条件
                    print('####################################converge iteration:', k )
                    print('converge in para updates',F_pnew)
                    break
                else:
                    p_new=p.clone()
                    p_new[selected_columns]+= alpha * torch.squeeze(h_lm)
                                        
                    F_p = F_fun(fx)                    
                    fx_new = fx_fun(p_new)
                    F_pnew = F_fun(fx_new)             
                    o = F_p - F_pnew
                    o_=torch.matmul(gkF.t(),h_lm)+1/2*torch.matmul(h_lm.t(),torch.matmul(A_opt,h_lm))+1/2*mu*torch.norm(h_lm, p=2)
                    # print('gkF',torch.norm(h_lm,p=2)**2)
                    if o/o_ > yi and torch.norm(gkF,p=2)**2>yi2/mu:
                        self.loss_record[self.loss_iter] = float(F_pnew.item())
                        self.loss_iter += 1
                        self.time_record[self.time_iter] = elapsed_time_ms/1000
                        self.time_iter += 1
                        if k%10==0:
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
                    
                    
    
        self.p=p     
        self.avg_time = np.sum(self.time_record[self.time_record != 0])     
                
        
    
    
    def plt(self):
        pass

    def error(self):
        pass        
                
    import torch

    def adam_step(self, itr, state=None, step_size=0.001, b1=0.9, b2=0.999, eps=1e-8):
            """
            Adam optimizer step as described in http://arxiv.org/pdf/1412.6980.pdf.
            Arguments:
            - g: gradients (PyTorch tensor)
            - x: parameters to be updated (PyTorch tensor)
            - itr: current iteration (integer)
            - state: tuple of (m, v) or None
            - step_size: learning rate (float)
            - b1: beta1 parameter (float)
            - b2: beta2 parameter (float)
            - eps: epsilon for numerical stability (float)
            - device: device to run on ('cpu' or 'cuda')
            """
            p_number = self.p_number
            params = torch.cat([p.view(-1) for p in self.model.parameters()], dim=0).to(device) # 把模型参数展平成一维向量
            x=params.to(device)
            def f(params, input_data):
            
                with torch.no_grad():
                    a = self.hidden_size1 * self.input_size
                    self.model.model.layer1.weight.data = params[:a].reshape(self.hidden_size1, self.input_size).clone()  # layer1的权重
                    self.model.model.layer1.bias.data = params[a:a + self.hidden_size1].clone()  # layer1的偏置
                    a += self.hidden_size1
                    self.model.model.layer2.weight.data = params[a:a + self.hidden_size1 * self.hidden_size2].reshape(self.hidden_size2, self.hidden_size1).clone()  # layer2的权重
                    a += self.hidden_size1 * self.hidden_size2
                    self.model.model.layer2.bias.data = params[a:a + self.hidden_size2].clone()  # layer2的偏置
                    a += self.hidden_size2
                    self.model.model.layer3.weight.data = params[a:a + self.hidden_size2 * self.hidden_size2].reshape(self.hidden_size2, self.hidden_size2).clone()  # layer2的权重
                    a += self.hidden_size2 * self.hidden_size2
                    self.model.model.layer3.bias.data = params[a:a + self.hidden_size2].clone()  # layer2的偏置
                    a += self.hidden_size2
                    self.model.model.layer4.weight.data = params[a:a + self.output_size * self.hidden_size2].reshape(self.output_size, self.hidden_size2).clone()  # layer3的权重
                    a += self.output_size * self.hidden_size2
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
                
            
                fx = du_dyy+du_dxx + 68* torch.sin(8*self.X_inside[:,0])*torch.sin(2*self.X_inside[:,1])
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
                    with torch.no_grad():
                        f_inside = f(params, input)  # 更换模型中的参数
                    
                    func_model, func_params = make_functional(self.model)
                    def fm(x, func_params):
                        fx = func_model(func_params, x)
                        return fx.squeeze(0).squeeze(0)
                    def floss(func_params,input):
                        f_inside=fm(input, func_params)        
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
            
            if state is None:
                    m = torch.zeros_like(x)
                    v = torch.zeros_like(x)
            else:
                    m, v = state
            for i in range(itr):
                fx=fx_fun(x)
                J=J_func(x)
                g = torch.matmul(J.t(), fx)
                # Move tensors to the specified device
                if i%10 ==0:
                    print('step:',i,'loss:',F_fun(fx))
                
                # Initialize state if not provided
                
                with torch.no_grad():
                    # Update biased first moment estimate
                    m = b1 * m + (1 - b1) * g
                    # Update biased second raw moment estimate
                    v = b2 * v + (1 - b2) * (g ** 2)
                    
                    # Compute bias-corrected first moment estimate
                    mhat = m / (1 - b1 ** (itr + 1))
                    # Compute bias-corrected second raw moment estimate
                    vhat = v / (1 - b2 ** (itr + 1))
                    
                    # Update parameters
                    x = x - (step_size * mhat) / (torch.sqrt(vhat) + eps)
            fx=fx_fun(x)
            print('adam end up loss : ',F_fun(fx))
            torch.save(self.model.state_dict(), 'model.pth')
            

            
            

        

        
        
    
    
    def plt(self):
        pass

    def error(self):
        pass
    







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
        