"""PINN_LM"""
import torch
import torch.nn as nn
import collections 
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
    """
    同为2 维 : 需要修改的方法：
    newspampling()
    LM(): fx_fun() 
          J_func()
    
          plot_l2error.py 中exact solution
    """
    def __init__(self,cuda_num=1):
        self.input_size = 2
        self.hidden_size1=100#100
        self.hidden_size2=100
        self.output_size = 1
        self.depth = 2
        self.device=torch.device(f"cuda:{cuda_num}") if torch.cuda.is_available() else torch.device("cpu")  # 选择使用GPU还是CPU
        global device
        device = self.device
        self.model = Network(self.input_size, self.hidden_size1, self.hidden_size2,    self.output_size, self.depth, act=  torch.nn.Tanh() ).double().to(device)  # 定义神经网络
        self.p_number = self.model.p_number # 参数的个数
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
            random_samples = 40  #square 
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
            
            random_samples = 100 #*4
            torch.manual_seed(4+i)
            # y = (upper_boundy - lower_boundy) * torch.rand(random_samples) + lower_boundy 
            y = torch.linspace(lower_boundy, upper_boundy, random_samples)
            bc1 = torch.stack(torch.meshgrid(torch.tensor(lower_boundx).double(), y)).reshape(2, -1).T  # x=-1边界
            bc2 = torch.stack(torch.meshgrid(torch.tensor(upper_boundx).double(), y)).reshape(2, -1).T  # x=+1边界
            random_samples=100
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

    
    def LM(self,opt_num, step,modify=True,bd_tol=0):
        """ 
        Parameters:
        opt_num -  优化的参数数量 
        step: LM 下降的次数 
        deterministic: 是否要严格下降 True 为是
        mu:  damp parameter >0 为LM method， 0 为Gauss-Newton => deterministic=False
        需要修改的部分：fx_fun 函数; J_func 函数; 
        """

        
        p_number = self.p_number
        params = torch.cat([p.view(-1) for p in self.model.parameters()], dim=0).to(device) # 把模型参数展平成一维向量

        
    
        def f(params, input_data):
                a = 0
                with torch.no_grad():
                    for name, module in self.model.model.named_children():            
                        if isinstance(module, nn.Linear):
                            in_features = module.weight.size(1)
                            out_features = module.weight.size(0)
                            weight_size = in_features * out_features
                            # 权重初始化
                            module.weight.data = params[a:a + weight_size].reshape(out_features, in_features).clone()
                            a += weight_size
                            # 偏置初始化（如果有）
                            if module.bias is not None:
                                module.bias.data = params[a:a + out_features].clone()
                                a += out_features
                
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
        def broyden_partial_update(J_prev, delta_r, delta_theta, selected_indices):
            """
            对 Jacobian 的部分列做 Broyden 更新

            参数:
                J_prev: torch.Tensor, shape (m, n) -- 上一步的雅可比矩阵
                delta_r: torch.Tensor, shape (m,) -- 残差变化
                delta_theta: torch.Tensor, shape (k,) -- 参数子集的变化
                selected_indices: list[int] or np.ndarray -- 当前更新参数对应的索引，长度为 k

            返回:
                J_new: torch.Tensor, shape (m, n)
            """
            with torch.no_grad():
                J_new = J_prev.clone()
                
                # 取出被更新的列
                s=1
                J_sub = J_prev[:, selected_indices]  # shape: (m, k)
                
                # 计算更新项
                correction = torch.outer((delta_r - s*torch.matmul(J_sub ,delta_theta)), delta_theta )/ s*(delta_theta.T @ delta_theta)

                # 更新 J 中的部分列
                J_new[:, selected_indices] = J_sub + correction

            return J_new
        
        #参数准备
        mu = 1000
        lmin, lmax = 1e-16, 1e100
        lambda_up, lambda_down = 2, 0.5
        yi, yi2 = 1e-16, 1e-6
        alpha = 1
        F_pnew = 1
        k, reject = 0, 0
        kmax = step
        p = params.to(device)
        J = J_func(p)
        A = torch.matmul(J.t(),J)
        F_p = torch.tensor(10).to(device)
        diag = torch.eye(p_number)
        fx = fx_fun(p)
        elapsed_time_ms=0
        ####随机选择部分参数进行优化
        selected_columns = np.random.choice(p_number, opt_num, replace=False)
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
                
                J_opt=J[:,selected_columns]
                
                A_opt= torch.matmul(J_opt.t(),J_opt)
                diag=torch.eye(A_opt.shape[0]).to(device)
                H = A_opt + mu * diag
                
                fx = fx_fun(p)
                
                with torch.no_grad():
                    
                    gk=torch.matmul(J.t(),fx)
                    gkF=gk[selected_columns]
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
                    p_new[selected_columns]+= alpha * torch.squeeze(h_lm)
                                        
                    F_p = F_fun(fx)        
                    fx_new = fx_fun(p_new)
                    F_pnew = F_fun(fx_new)             
                    o = F_p - F_pnew
                    o_=torch.matmul(gkF.t(),h_lm)+1/2*torch.matmul(h_lm.t(),torch.matmul(A_opt,h_lm))+1/2*mu*torch.norm(h_lm, p=2)
                    
                    
                    ratio=o/o_
                   
                    if ratio > yi and torch.norm(gkF,p=2)**2>yi2/mu:
                        reject=0
                        self.loss_record[self.loss_iter] = float(F_pnew.item())
                        self.loss_iter += 1
                        self.time_record[self.time_iter] = elapsed_time_ms/1000
                        self.time_iter += 1
                        
                        if k%1==0:
                            print(f"steps {k} accept move | Loss = {F_p.item()}, Loss new = {F_pnew.item()} | parameter #: {opt_num} | Elapsed: {elapsed_time_ms:.1f}ms") 
                        
                        r=fx_new-fx
                        p = p_new
                        
                        #update J
                        if reject >= bd_tol:
                            J = J_func(p)
                        else:
                            J = broyden_partial_update(J, r, h_lm.view(-1), selected_columns)
                                # J=J+(1/delta.pow(2).sum())* torch.outer((()-torch.matmul(J,delta)), delta)

                        
                        mu = max(mu * lambda_down,lmin)  
                        
                        if k%5==0:
                            rho=1/(2*torch.norm(gk,p=2)**4 * mu**2)
                            if rho <=1 :
                                opt_num=int(np.round(self.p_number*torch.sqrt(1-rho).item()))
                                print("optnumber",opt_num)
                                #torch.sqrt(1-1/(2*torch.norm(gk,p=2)**4 * mu**2))
                            else:
                                print('zone')
                                opt_num=int(np.round(2/3*p_number))
                                
                            selected_columns = np.random.choice(p_number, opt_num, replace=False)
                        
                    else:
                        if k % 1 == 0:print("reject move")
                        if reject >= bd_tol:
                            J = J_func(p)
                        reject+=1
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
                    
                    gk=torch.matmul(J.t(),fx)
                    gkF=gk[selected_columns]
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
                        reject=0
                        self.loss_record[self.loss_iter] = float(F_pnew.item())
                        self.loss_iter += 1
                        self.time_record[self.time_iter] = elapsed_time_ms/1000
                        self.time_iter += 1
                        if k%1==0:
                            print(f"steps {k} accept move | Loss = {F_p.item()}, Loss new = {F_pnew.item()} | Elapsed: {elapsed_time_ms:.1f}ms") 
                        
                        p = p_new
                        r=fx_new-fx
                        # J = J_func(p)  # update J
                        
                        if reject >= bd_tol:
                            J = J_func(p)
                        else:
                            J = broyden_partial_update(J, r, h_lm.view(-1), selected_columns)
                        mu = max(mu * lambda_down,lmin)  # lower limit u =1e-11
                    else:
                        if reject >= bd_tol:
                            J = J_func(p)
                        if k % 1 == 0:print("reject move")
                        reject+=1
                        mu = min(mu * lambda_up,lmax)  # Up limit u =1e11
                    end_event.record()
                    torch.cuda.synchronize()  # Wait for the events to be recorded!
                    elapsed_time_ms = start_event.elapsed_time(end_event)
                    
                    

        self.time_record[self.time_iter] = elapsed_time_ms/1000
        self.loss_record[self.loss_iter] = float(F_pnew.item())
        self.p=p     
        self.avg_time = np.sum(self.time_record[self.time_record != 0])     
                
        
    
    
    def plt(self):
        pass

    def error(self):
        pass        
                
    

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
                a = 0
                with torch.no_grad():
                    for name, module in self.model.model.named_children():            
                        if isinstance(module, nn.Linear):
                            in_features = module.weight.size(1)
                            out_features = module.weight.size(0)
                            weight_size = in_features * out_features
                            # 权重初始化
                            module.weight.data = params[a:a + weight_size].reshape(out_features, in_features).clone()
                            a += weight_size
                            # 偏置初始化（如果有）
                            if module.bias is not None:
                                module.bias.data = params[a:a + out_features].clone()
                                a += out_features
                
                
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
                J = torch.zeros(self.X_inside_num+self.X_boundary_num, p_number).to(device)
                params.requires_grad_(True)
                def Inter(params, input):
                    with torch.no_grad():
                        f_inside = f(params, input) 
                    
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
            
    def lbfgs_step(self, max_iter=100, history_size=100, gtol=1e-10):
            

            device = self.device
            p_number = self.p_number
            x = torch.cat([p.view(-1) for p in self.model.parameters()], dim=0).detach().clone().to(device).requires_grad_(True)

            s_list, y_list, rho_list = [], [], []


            # f: 用当前参数生成模型预测
            def f(params, input_data):
                a = 0
                with torch.no_grad():
                    for module in self.model.model.children():
                        if isinstance(# The above code is not doing anything as it consists of only
                        # comments (lines starting with #). It does not contain any
                        # executable code.
                        module, nn.Linear):
                            in_features = module.weight.size(1)
                            out_features = module.weight.size(0)
                            w_len = in_features * out_features
                            module.weight.data = params[a:a + w_len].reshape(out_features, in_features).clone()
                            a += w_len
                            if module.bias is not None:
                                module.bias.data = params[a:a + out_features].clone()
                                a += out_features
                return torch.vmap(self.model)(input_data)

            # 残差函数
            def fx_fun(params):
                f_inside = f(params, self.X_inside)
                du_dX = torch.autograd.grad(f_inside, self.X_inside, torch.ones_like(f_inside), retain_graph=True, create_graph=True)[0][:, 0]
                du_dY = torch.autograd.grad(f_inside, self.X_inside, torch.ones_like(f_inside), retain_graph=True, create_graph=True)[0][:, 1]
                du_dxx = torch.autograd.grad(du_dX, self.X_inside, torch.ones_like(du_dX), retain_graph=True, create_graph=True)[0][:, 0]
                du_dyy = torch.autograd.grad(du_dY, self.X_inside, torch.ones_like(du_dY), retain_graph=True, create_graph=True)[0][:, 1]
                fx_in = du_dxx + du_dyy + 68 * torch.sin(8 * self.X_inside[:, 0]) * torch.sin(2 * self.X_inside[:, 1])
                f_bd = f(params, self.X_boundary)
                return torch.cat([fx_in.view(-1), f_bd.view(-1)], dim=0)

            def F_fun(fx):
                return torch.mean(fx ** 2)

            # 计算雅可比矩阵
            def J_func(params):
                J = torch.zeros(self.X_inside_num + self.X_boundary_num, p_number).to(device)

                def Inter(params, input):
                    with torch.no_grad():
                        f_inside = f(params, input) 
                    func_model, func_params = make_functional(self.model)
                    def fm(x, func_params):
                        fx = func_model(func_params, x)
                        return fx.squeeze(0).squeeze(0)
                    def floss(func_params, input):
                        d2u = jacrev(jacrev(lambda x: fm(x, func_params)))(input)
                        du_dxx, du_dyy = d2u[0][0], d2u[1][1]
                        return du_dxx + du_dyy + 68 * torch.sin(8 * input[0]) * torch.sin(2 * input[1])
                    grads = vmap(jacrev(floss), (None, 0))(func_params, input)
                    return torch.hstack([g.reshape(len(g), -1) for g in grads]).detach()

                def Bound(params, input):
                    with torch.no_grad():
                        f_inside = f(params, input) 
                    func_model, func_params = make_functional(self.model)
                    def fm(x, func_params):
                        fx = func_model(func_params, x)
                        return fx.squeeze(0).squeeze(0)
                    def floss(func_params, input):
                        return fm(input, func_params)
                    grads = vmap(jacrev(floss), (None, 0))(func_params, input)
                    return torch.hstack([g.reshape(len(g), -1) for g in grads]).detach()

                J[:self.X_inside_num, :] = Inter(params, self.X_inside)
                J[self.X_inside_num:, :] = Bound(params, self.X_boundary)
                return J

            def compute_grad(params, fx):
                J = J_func(params)
                return torch.matmul(J.T, fx.detach())

            def two_loop_recursion(g):
                q = g.detach().clone()
                alpha = []
                for i in reversed(range(len(s_list))):
                    s, y, rho = s_list[i], y_list[i], rho_list[i]
                    a = rho * torch.dot(s, q)
                    alpha.append(a)
                    q -= a * y
                if len(s_list) > 0:
                    gamma = torch.dot(s_list[-1], y_list[-1]) / (torch.dot(y_list[-1], y_list[-1]) + 1e-10)
                else:
                    gamma = 1.0
                r = gamma * q
                for i in range(len(s_list)):
                    s, y, rho = s_list[i], y_list[i], rho_list[i]
                    b = rho * torch.dot(y, r)
                    r += s * (alpha[-(i + 1)] - b)
                return -r

            # 初始化
            elapsed_time_ms=0
            start_event = torch.cuda.Event(enable_timing=True)
            step_event=torch.cuda.Event(enable_timing=True)
            endstep=torch.cuda.Event(enable_timing=True) #
            start_event.record()
            fx = fx_fun(x)
            g = compute_grad(x, fx)
            
            for i in range(max_iter):
                end_event = torch.cuda.Event(enable_timing=True)
                if g.norm() < gtol:
                    print(f"[L-BFGS] Early stop at iter {i}, grad norm: {g.norm().item():.2e}")
                    break
                p = two_loop_recursion(g)
                
                # 加入 Armijo 线搜索
                alpha = 1
                c = 1e-5
                while True:
                    x_new = x + alpha * p
                    fx_new = fx_fun(x_new)
                    if F_fun(fx_new) <= F_fun(fx) + c * alpha * torch.dot(g, p):
                        break
                    alpha *= 0.7
                    if alpha < 1e-10:
                        print(f"[Iter {i}] Line search failed. Stop.")
                        return
                
            
                g_new = compute_grad(x_new, fx_new)
                s = x_new - x
                y = g_new - g
                rho = 1.0 / (torch.dot(y, s) + 1e-15)
                if len(s_list) == history_size:
                    s_list.pop(0)
                    y_list.pop(0)
                    rho_list.pop(0)
                s_list.append(s.detach())
                y_list.append(y.detach())
                rho_list.append(rho)

                with torch.no_grad():
                    x = x_new.detach().clone().to(device)
                x.requires_grad_(True)
                fx = fx_new
                g = g_new
                F_pnew=F_fun(fx)
                if i % 5 == 0:
                    print(f"[L-BFGS] Iter {i}, Loss: {F_pnew:.6e}, Grad Norm: {g.norm():.2e},Elapsed: {elapsed_time_ms:.1f}ms")
                self.loss_record[self.loss_iter] = float(F_pnew)
                self.loss_iter += 1
                self.time_record[self.time_iter] = elapsed_time_ms/1000
                self.time_iter += 1
                end_event.record()
                torch.cuda.synchronize()  # Wait for the events to be recorded!
                elapsed_time_ms = start_event.elapsed_time(end_event)

            print("[L-BFGS] Final Loss:", F_fun(fx).item())
            with torch.no_grad():
                f_inside = f(x, self.X_boundary) 
            torch.save(self.model.state_dict(), "model_lbfgs.pth")   
            self.avg_time = np.sum(self.time_record[self.time_record != 0])    
            

            
            

        

        
        
    
    






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
            # 检查 depth 参数
            assert depth <= 3, "depth 初始化无效，depth 必须小于等于 5"
            
            # 根据 depth 动态构建模型
            layers = []

            layers.append(("layer1", nn.Linear(input_size, hidden_size1)))
            layers.append(("activation1", act))

            # 第1层
            
            layers.append(("layer2", nn.Linear(hidden_size1, output_size,bias=False)))
            
            if depth>=2:
                layers[2] = ("layer2", nn.Linear(hidden_size1, hidden_size2))
                layers.append(("activation2", act))
                layers.append(("layer3", nn.Linear(hidden_size2, output_size,bias=False)))
                

            if depth >=3:
                layers[4]=("layer3", nn.Linear(hidden_size2, hidden_size2))
                layers.append(("activation3", act))
                layers.append(("layer4", nn.Linear(hidden_size2, output_size, bias=False)))
            
            self.model = nn.Sequential(collections.OrderedDict(layers))
            self.p_number = self.calculate_total_params()
            print("parameter num",self.p_number)
            # assert depth<=5, "depth 初始化无效"
            # if depth==5:
            #     self.model = nn.Sequential(
            #         collections.OrderedDict
            #             (
            #             [("layer1",nn.Linear(in_features=input_size, out_features=hidden_size1)),
            #         ("activation1",act),
            #         ("layer2",nn.Linear(in_features=hidden_size1, out_features=hidden_size2)),
            #         ("activation2",act),
            #         ("layer3",nn.Linear(in_features=hidden_size2, out_features=hidden_size2)),
            #         ("activation3",act),
            #         ("layer4",nn.Linear(in_features=hidden_size2, out_features=output_size, bias=False))]
            #         )
                    
            #     )
            # else:
            #     self.model = nn.Sequential(
            #         collections.OrderedDict
            #             (
            #             [("layer1",nn.Linear(in_features=input_size, out_features=hidden_size1)),
            #         ("activation1",act),
            #         ("layer2",nn.Linear(in_features=hidden_size1, out_features=output_size, bias=False))]
            #         )
            #      ）
                
        def calculate_total_params(self):
                """ 动态计算参数总数 """
                total_params = 0
                for name, module in self.model.named_children():
                    if isinstance(module, nn.Linear):
                        # 计算权重参数数量
                        total_params += module.weight.numel()
                        # 计算偏置参数数量（如果有偏置）
                        if module.bias is not None:
                            total_params += module.bias.numel()
                return total_params
        def forward(self, x):
            x = self.model(x)
            return x
        