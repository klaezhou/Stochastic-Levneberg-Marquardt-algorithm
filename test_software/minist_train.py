import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import make_functional, vmap, grad, jacrev, hessian
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(0),  
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=0)  
        )
        self.p_number = self.calculate_total_params()

    def forward(self, x):
        return self.model(x)  # 直接调用 Sequential

    def calculate_total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    
def LM(net,input_data,target,opt_num,step,modify=True,cuda_num=2):
        loss_record=np.zeros(100000)
        loss_iter=0
        time_iter=0
        time_record=np.zeros(100000)
        device = torch.device(f"cuda:{cuda_num}") if torch.cuda.is_available() else torch.device("cpu")  # 选择使用GPU还是CPU
        p_number = net.p_number
        params = torch.cat([p.view(-1) for p in net.parameters()], dim=0).to(device) # 把模型参数展平成一维向量
        data_num=input_data.shape[0]
        def nll_loss(output):
            loss = F.nll_loss(output, target)
            return loss
        def f(params, input_data):
            a = 0
            with torch.no_grad():
                for name, module in net.model.named_children():  # 适用于 nn.Sequential
                    if isinstance(module, nn.Linear):
                        in_features = module.weight.size(1)
                        out_features = module.weight.size(0)
                        weight_size = in_features * out_features
                        module.weight.data = params[a:a + weight_size].reshape(out_features, in_features).clone()
                        a += weight_size
                        if module.bias is not None:
                            module.bias.data = params[a:a + out_features].clone()
                            a += out_features
                    elif isinstance(module, nn.Conv2d):
                        out_channels, in_channels, kernel_h, kernel_w = module.weight.shape
                        weight_size = out_channels * in_channels * kernel_h * kernel_w
                        module.weight.data = params[a:a + weight_size].reshape(out_channels, in_channels, kernel_h, kernel_w).clone()
                        a += weight_size
                        if module.bias is not None:
                            module.bias.data = params[a:a + out_channels].clone()
                            a += out_channels

            model_output = torch.vmap(net, randomness='different')(input_data)
            return model_output
        
        def J_func(params)->torch.tensor:
            J=torch.zeros(data_num,p_number).to(device)
            params.requires_grad_(True)
            def Diff(params,input_data):
                with torch.no_grad():
                    f_output=f(params,input_data)
                
                func_model, func_params = make_functional(net)
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
            J=Diff(params,input_data)
            
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
        output = f(p,input_data)
        mu=1000
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
                
                output= f(p)
                
                with torch.no_grad():
                    gkF=torch.matmul(J_opt.t(),fx)
                    gk=torch.matmul(J.t(),fx)
                
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
                                        
                    F_p = nll_loss(output)   
                                  
                    output_new = f(p_new)
                      
                    F_pnew = nll_loss(output_new)             
                    o = F_p - F_pnew
                     
                    o_=torch.matmul(gkF.t(),h_lm)+1/2*torch.matmul(h_lm.t(),torch.matmul(A_opt,h_lm))+1/2*mu*torch.norm(h_lm, p=2)
                    
                    
                    ratio=o/o_
                   
                    if ratio > yi and torch.norm(gkF,p=2)**2>yi2/mu:
                        loss_record[loss_iter] = float(F_pnew.item())
                        loss_iter += 1
                        time_record[time_iter] = elapsed_time_ms/1000
                        time_iter += 1
                        
                        if k%1==0:
                            print(f"steps {k} accept move | Loss = {F_p.item()}, Loss new = {F_pnew.item()} | Elapsed: {elapsed_time_ms:.1f}ms") 
                        p = p_new
                        
                        J = J_func(p)  # update J
                        
                        mu = max(mu * lambda_down,lmin)  
                        
                        if k%1==0:
                            if 1/(2*torch.norm(gk,p=2)**4 * mu**2) <=1 :
                                opt_num=int(np.round(p_number*torch.sqrt(1-1/(2*torch.norm(gk,p=2)**4 * mu**2)).item()))
                                print("optnumber",opt_num)
                                #torch.sqrt(1-1/(2*torch.norm(gk,p=2)**4 * mu**2))
                            else:
                                print('zone')
                                opt_num=int(np.round(1/3*p_number))
                                
                            selected_columns = np.random.choice(p_number, opt_num, replace=False)
                        
                    else:
                        if k % 1 == 0:print("reject move")
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
                
                fx = f(p)
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
                        loss_record[loss_iter] = float(F_pnew.item())
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
                        J = J_func(p)  # update J
                        mu = max(mu * lambda_down,lmin)  # lower limit u =1e-11
                    else:
                        if k % 1 == 0:print("reject move")
                        mu = min(mu * lambda_up,lmax)  # Up limit u =1e11
                    end_event.record()
                    torch.cuda.synchronize()  # Wait for the events to be recorded!
                    elapsed_time_ms = start_event.elapsed_time(end_event)
                    
                    
        
        avg_time = np.sum(time_record[time_record != 0])     
                    
            
            
            
            
            
            
        
def main():
    torch.set_default_dtype(torch.float64)
    #载入和初始化数据
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST(root="~/.torch/MNIST_data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST(root="~/.torch/MNIST_data", train=False, transform=transform)
    
    #转换为特征矩阵
    X_train = np.array([np.array(image).reshape(-1) for image, label in dataset1]) 
    y_train = np.array([label for image, label in dataset1])  
    print("X_train:",X_train.shape)
    X_test = np.array([np.array(image).reshape(-1) for image, label in dataset2]) 
    y_test = np.array([label for image, label in dataset2])  
    
    # 将 NumPy 数组转换为 PyTorch Tensor，并调整形状
    X_train = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 1, 28, 28)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_test = torch.tensor(X_test, dtype=torch.float32).reshape(-1, 1, 28, 28)
    y_test = torch.tensor(y_test, dtype=torch.long)
    print(X_train.shape[0])
    # 输出形状
    print("X_train:", X_train.shape)  # (60000, 1, 28, 28)
    print("y_train:", y_train.shape)  # (60000,)
    print("X_test:", X_test.shape)    # (10000, 1, 28, 28)
    print("y_test:", y_test.shape)    # (10000,)
    print("y_trian",y_train)
    model=Net()
    print(model.p_number)
    
    
    
    
if __name__ == '__main__':
    main()