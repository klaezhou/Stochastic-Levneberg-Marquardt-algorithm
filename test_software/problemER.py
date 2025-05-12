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
import time
from matplotlib.ticker import FuncFormatter
pi=np.pi
class PINN_LM:
    def __init__(self):
        
        self.hidden_size =500
        torch.set_default_dtype(torch.float64)
        global device
        device = torch.device("cuda:7") if torch.cuda.is_available() else torch.device("cpu")  # 选择使用GPU还是CPU
        self.p=torch.zeros(self.hidden_size).to(device)
        self.p[1::2]-=1.2
        self.p[::2]+=1
        

        self.p_number = self.hidden_size
        print('model # of parameters',self.p_number)
        
        
        
        self.loss_record=np.zeros(100000)
        self.loss_iter=0
        self.Sampledata=np.array([])
        self.time_iter=0
        self.time_record=np.zeros(100000)

        
    
    

    
    def LM(self,opt_num, step,modify=True):
        """ 
        Parameters:
        opt_num -  优化的参数数量 
        step: LM 下降的次数 
        deterministic: 是否要严格下降 True 为是
        mu:  damp parameter >0 为LM method， 0 为Gauss-Newton => deterministic=False
        需要修改的部分：fx_fun 函数; J_func 函数; 
        """

        
        p_number = self.p_number
        params = self.p


        import torch

        import torch

        def F11(x):
            n = x.size(0)
            f_i = torch.zeros(n, device=device)
            f_i[1::2] += 10*(x[::2] - x[0:n//2])
            f_i[::2] += 1-x[1::2]
            # 最后减去 1
            

            

            return f_i

        
        


        def F_fun(fx)->torch.tensor:
            '''计算L2范数'''
            F_p = 0
            for i in range(fx.size(0)):
                F_p += (fx[i]) ** 2 
            F_p /= fx.size(0)
            return F_p
        

        def J_func(params)->torch.tensor:
            J = torch.zeros(self.hidden_size, p_number).to(device)# 初始化雅可比矩阵
            params.requires_grad_(True)
            
            J= torch.autograd.functional.jacobian(F11, params)
            
            
            
            
            return J
        
        
        #参数准备
        lmin=1e-15
        lmax=1e100
        k = 0
        kmax = step
        p = params.to(device)
        J = J_func(p)
        A = torch.matmul(J.t(),J)
        F_p = torch.tensor(10).to(device)
        F_pnew = 1
        alpha = 1
        lambda_up = 2
        lambda_down = 0.5
        yi=1e-15
        yi2=1e-15
        diag = torch.eye(p_number)
        fx = F11(p)
        mu=1
        elapsed_time_ms=0
        ####随机选择部分参数进行优化
        selected_columns = np.random.choice(p_number, opt_num, replace=False)
        ####
        if modify:
            print('begin a new iteration')
            '''严格lm'''
            while (k < kmax):  
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                k = k + 1
                
                J_opt=J[:,selected_columns]
                
                A_opt= torch.matmul(J_opt.t(),J_opt)
                diag=torch.eye(A_opt.shape[0]).to(device)
                H = A_opt + mu * diag
                
                fx = F11(p)
                with torch.no_grad():
                    gkF=torch.matmul(J_opt.t(),fx)
                    gk=torch.matmul(J.t(),fx)
                try:              
                    h_lm = torch.linalg.solve(H, -gkF)             
                except:
                    print('singular matrix')
                
                if  F_pnew<1e-10:  # 满足收敛条件
                    print('###########################################converge iteration:',k)
                    print('converge in para updates')
                    break
                else:
                    p_new=p.clone()
                    p_new[selected_columns]+= alpha * torch.squeeze(h_lm)
                                        
                    F_p = F_fun(fx)                    
                    fx_new = F11(p_new)
                    F_pnew = F_fun(fx_new)             
                    o = F_p - F_pnew
                    o_=torch.matmul(gkF.t(),h_lm)+1/2*torch.matmul(h_lm.t(),torch.matmul(A_opt,h_lm))+1/2*mu*torch.norm(h_lm, p=2)
                    # print('gkF', torch.norm(gkF,p=2)**2)
                    # if o ==0:
                    #     print('o==0')
                    #     continue
                    # print('yi2/mu',yi2/mu)
                    ratio=o/o_
                    # ratio=0.01 if ratio==0 or torch.isnan(ratio) else ratio
                    # print(torch.norm(gkF,p=2)**2>=yi2/mu) 
                    if ratio > yi and torch.norm(gkF,p=2)**2>yi2/mu:
                        self.loss_record[self.loss_iter] = float(F_pnew.item())
                        self.loss_iter += 1
                        
                        if k%10==0:
                            print("steps ", k, end=' ')
                            print('accept move')
                            print("Loss =", F_p.item(), end=' ')
                            print("Loss new=", F_pnew.item())
                            print(f'Elapsed: {elapsed_time_ms:.1f}ms')  
                        p = p_new
                        J = J_func(p)  # update J
                        mu = max(mu * lambda_down,lmin)  
                        if k%1==0:
                            if 1/(2*torch.norm(gk,p=2)**4 * mu**2) <=1 :
                                opt_num=int(np.round(p_number*torch.sqrt(1-1/(2*torch.norm(gk,p=2)**4 * mu**2)).item()))
                                print('Zhuyi:',opt_num) #torch.sqrt(1-1/(2*torch.norm(gk,p=2)**4 * mu**2))
                            else:
                                opt_num=int(np.round(2/3*p_number))
                                
                                print('Zhuyi:', opt_num)
                            selected_columns = np.random.choice(p_number, opt_num, replace=False)
                        
                    else:
                        if k % 10 == 0:print("reject move")
                        mu = min(mu * lambda_up,lmax)  
                    end_event.record()
                    torch.cuda.synchronize()  # Wait for the events to be recorded!
                    elapsed_time_ms = start_event.elapsed_time(end_event)
                    self.time_record[self.time_iter] = elapsed_time_ms
                    self.time_iter += 1
                    
                
        else:
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
                
                fx = F11(p)
                gkF=torch.matmul(J_opt.t(),fx)
                gk=torch.matmul(J.t(),fx)
                try:              
                    h_lm = torch.linalg.solve(H, -gkF)             
                except:
                    print('singular matrix')
                
                if  F_pnew<1e-10:  # 满足收敛条件
                    print('####################################converge iteration:', k )
                    print('converge in para updates',F_pnew)
                    break
                else:
                    p_new=p.clone()
                    p_new[selected_columns]+= alpha * torch.squeeze(h_lm)
                                        
                    F_p = F_fun(fx)                    
                    fx_new = F11(p_new)
                    F_pnew = F_fun(fx_new)             
                    o = F_p - F_pnew
                    o_=torch.matmul(gkF.t(),h_lm)+1/2*torch.matmul(h_lm.t(),torch.matmul(A_opt,h_lm))+1/2*mu*torch.norm(h_lm, p=2)
                    # print('gkF',torch.norm(h_lm,p=2)**2)
                    if o/o_ > yi and torch.norm(gkF,p=2)**2>yi2/mu:
                        self.loss_record[self.loss_iter] = float(F_pnew.item())
                        self.loss_iter += 1
                        
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
                    self.time_record[self.time_iter] = elapsed_time_ms
                    self.time_iter += 1
                    
    
        self.p=p     
        self.avg_time = np.sum(self.time_record[self.time_record != 0])        
                
        
    def plt(self):
        pass

    def error(self):
        pass
    def distance(self):
        print('dis:',torch.norm(self.p-torch.ones(self.hidden_size),p=2))
    
    





    

