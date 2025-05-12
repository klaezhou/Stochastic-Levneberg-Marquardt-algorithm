import torch
import torch.nn as nn
import numpy as np
from functorch import make_functional, vmap, jacrev
# from ucimlrepo import fetch_ucirepo 
import collections
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd


class Network(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, depth, act=torch.nn.Sigmoid()):
        super(Network, self).__init__()
        layers = []
        layers.append(("layer1", nn.Linear(input_size, hidden_size1)))
        layers.append(("activation1", act))
        if depth >= 2:
            layers.append(("layer2", nn.Linear(hidden_size1, hidden_size2)))
            layers.append(("activation2", act))
            layers.append(("layer3", nn.Linear(hidden_size2, output_size, bias=False)))
        else:
            layers.append(("layer2", nn.Linear(hidden_size1, output_size, bias=False)))
        self.model = nn.Sequential(collections.OrderedDict(layers))
        self.p_number = sum(p.numel() for p in self.model.parameters())

    def forward(self, x):
        return self.model(x)

class PINN_LM:
    def __init__(self, X_data, y_data, cuda_num=6):
        self.device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
        self.X_data = X_data.to(self.device)
        self.y_data = y_data.to(self.device).view(-1)
        self.data_num = self.X_data.shape[0]
        torch.set_default_dtype(torch.float64)
        self.model = Network(self.X_data.shape[1],100, 100, 1, 2).double().to(self.device)
        self.p_number = self.model.p_number

        print('model # of parameters:', self.p_number)

        self.loss_record = np.zeros(100000)
        self.time_record = np.zeros(100000)
        self.loss_iter = 0
        self.time_iter = 0
        ## Initial weights
        def initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()
                        # m.bias.data.normal_(mean=0.0, std=1.0)

        initialize_weights(self.model)
        
    

    def f(self, params, input_data):
        a = 0
        for module in self.model.model.children():
            if isinstance(module, nn.Linear):
                in_f = module.weight.size(1)
                out_f = module.weight.size(0)
                w_size = in_f * out_f
                module.weight.data = params[a:a + w_size].reshape(out_f, in_f).clone()
                a += w_size
                if module.bias is not None:
                    module.bias.data = params[a:a + out_f].clone()
                    a += out_f
        return torch.vmap(self.model)(input_data)

    def fx_fun(self, params):
        preds = self.f(params, self.X_data).view(-1)
        return preds - self.y_data

    def F_fun(self, fx):
        return torch.mean(fx**2)

    def J_func(self, params):
        params.requires_grad_(True)
        func_model, func_params = make_functional(self.model)

        def fm(x, fp): return func_model(fp, x).squeeze()
        def floss(fp, x): return fm(x, fp)

        per_sample_grads = vmap(jacrev(floss), (None, 0))(func_params, self.X_data)
        
        cnt=0
                
        for g in per_sample_grads: 
            g = g.detach()
            J_d = g.reshape(len(g),-1) if cnt == 0 else torch.hstack([J_d,g.reshape(len(g),-1)])
            cnt = 1
        
        result= J_d.detach()
        return result

    def broyden_partial_update(self, J_prev, delta_r, delta_theta, indices):
        with torch.no_grad():
            J_sub = J_prev[:, indices]
            correction = torch.outer((delta_r - J_sub @ delta_theta), delta_theta) / (delta_theta.T @ delta_theta)
            J_new = J_prev.clone()
            J_new[:, indices] = J_sub + correction
        return J_new

    def SLM(self, opt_num=200, step=50,bd_tol=0,zone=2/3):
        p = torch.cat([p.view(-1) for p in self.model.parameters()]).to(self.device)
        J = self.J_func(p)
        mu = 1000
        lmin, lmax = 1e-16, 1e100
        lambda_up, lambda_down = 3, 0.5
        yi, yi2 = 1e-7, 1e-5
        alpha = 1
        F_pnew = 1
        k, reject = 0, 0
        elapsed_time_ms=0
        selected_cols = np.random.choice(self.p_number, opt_num, replace=False)
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        while k < step:
            end_event = torch.cuda.Event(enable_timing=True)
            k += 1
            fx = self.fx_fun(p)
            F_p = self.F_fun(fx)
            J_opt = J[:, selected_cols]
            A_opt =torch.matmul(J_opt.t(),J_opt)
            diag = torch.eye(A_opt.shape[0]).to(self.device)
            H = A_opt + mu * diag
            with torch.no_grad():
                    
                    gk=torch.matmul(J.t(),fx)
                    # gkF=torch.matmul(J_opt.t(),fx)
                    gkF=gk[selected_cols]
            try:              
                    h_lm = torch.linalg.solve(H, -gkF)      
                        
            except:
                    print('singular matrix')

            p_new = p.clone()
            p_new[selected_cols] += alpha * h_lm.view(-1)
            fx_new = self.fx_fun(p_new)
            F_pnew = self.F_fun(fx_new)
            o = F_p - F_pnew
            o_=torch.matmul(gkF.t(),h_lm)+1/2*torch.matmul(h_lm.t(),torch.matmul(A_opt,h_lm))+1/2*mu*torch.norm(h_lm, p=2)

            if o / o_ > yi and torch.norm(gkF)**2 > yi2 / mu:
                reject=0
                self.loss_record[self.loss_iter] = float(F_pnew.item())
                self.loss_iter += 1
                self.time_record[self.time_iter] = elapsed_time_ms/1000
                self.time_iter += 1
                print(f"[{k}] Accept | Loss = {F_p.item():.6f} -> {F_pnew.item():.6f} | parameter #: {opt_num}| time: {elapsed_time_ms}")
                p = p_new
                if reject >= bd_tol:
                    J = self.J_func(p)
                else:
                    J = self.broyden_partial_update(J, fx_new - fx, h_lm.view(-1), selected_cols)
                mu = max(mu * lambda_down, lmin)
                if k%5==0:
                    rho=1/(2*torch.norm(gk,p=2)**4 * mu**2)
                    if rho <=1 :
                        opt_num=int(np.round(self.p_number*torch.sqrt(1-rho).item()))
                        print("optnumber",opt_num)
                        #torch.sqrt(1-1/(2*torch.norm(gk,p=2)**4 * mu**2))
                    else:
                        print('zone')
                        opt_num=int(np.round(zone*self.p_number))
                        
                    selected_cols = np.random.choice(self.p_number, opt_num, replace=False)

            else:
                if reject >= bd_tol:
                    J = self.J_func(p)
                print(f"[{k}] Reject")
                reject += 1
                
                
                mu = min(mu * lambda_up, lmax)
                
            end_event.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            elapsed_time_ms = start_event.elapsed_time(end_event)
        
        self.time_record[self.time_iter] = elapsed_time_ms/1000
        self.loss_record[self.loss_iter] = float(F_pnew.item())
        print(f"Training finished. Final loss:{F_pnew.item()}, average time: {elapsed_time_ms/(1000*step)}")
        
        
        
        
        
        
        
        
    def LM(self, opt_num=200, step=50,bd_tol=0):
        p = torch.cat([p.view(-1) for p in self.model.parameters()]).to(self.device)
        J = self.J_func(p)
        mu = 1000
        lmin, lmax = 1e-16, 1e100
        lambda_up, lambda_down = 3, 0.5
        yi, yi2 = 1e-7, 1e-5
        alpha = 1
        F_pnew = 1
        k, reject = 0, 0
        elapsed_time_ms=0
        selected_cols = np.random.choice(self.p_number, opt_num, replace=False)
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        while k < step:
            end_event = torch.cuda.Event(enable_timing=True)
            k += 1
            fx = self.fx_fun(p)
            F_p = self.F_fun(fx)
            J_opt = J[:, selected_cols]
            A_opt =torch.matmul(J_opt.t(),J_opt)
            diag = torch.eye(A_opt.shape[0]).to(self.device)
            H = A_opt + mu * diag
            with torch.no_grad():
                    # gkF=torch.matmul(J_opt.t(),fx)
                    
                    gk=torch.matmul(J.t(),fx)
                    gkF=gk[selected_cols]
            try:              
                    h_lm = torch.linalg.solve(H, -gkF)      
                        
            except:
                    print('singular matrix')

            p_new = p.clone()
            p_new[selected_cols] += alpha * h_lm.view(-1)
            fx_new = self.fx_fun(p_new)
            F_pnew = self.F_fun(fx_new)
            o = F_p - F_pnew
            o_=torch.matmul(gkF.t(),h_lm)+1/2*torch.matmul(h_lm.t(),torch.matmul(A_opt,h_lm))+1/2*mu*torch.norm(h_lm, p=2)

            if o / o_ > yi and torch.norm(gkF)**2 > yi2 / mu:
                reject=0
                self.loss_record[self.loss_iter] = float(F_pnew.item())
                self.loss_iter += 1
                self.time_record[self.time_iter] = elapsed_time_ms/1000
                self.time_iter += 1
                print(f"[{k}] Accept | Loss = {F_p.item():.6f} -> {F_pnew.item():.6f} | parameter #: {opt_num}| time: {elapsed_time_ms}")
                p = p_new
                if reject >= bd_tol:
                    J = self.J_func(p)
                else:
                    J = self.broyden_partial_update(J, fx_new - fx, h_lm.view(-1), selected_cols)
                mu = max(mu * lambda_down, lmin)
                
            else:
                if reject >= bd_tol:
                    J = self.J_func(p)
                print(f"[{k}] Reject")
                reject += 1
                
                mu = min(mu * lambda_up, lmax)
            end_event.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            elapsed_time_ms = start_event.elapsed_time(end_event)
        self.time_record[self.time_iter] = elapsed_time_ms/1000
        self.loss_record[self.loss_iter] = float(F_pnew.item())

        print(f"Training finished. Final loss:{F_pnew.item()}, average time: {elapsed_time_ms/(1000*step)}")

def plot_loss_vs_time(loss1, time1, loss2, time2,loss3,time3,loss4,time4, label1="LM", label2="SLM",label3="LMB",label4="SLMB"):
        loss1 = loss1[time1 != 0]
        time1 = time1[time1 != 0]
        loss2 = loss2[time2 != 0]
        time2 = time2[time2 != 0]
        loss3 = loss3[time3!= 0]
        time3 = time3[time3 != 0]
        loss4 = loss4[time4 != 0]
        time4 = time4[time4 != 0]

        plt.figure(figsize=(10, 6))
        plt.plot(time1, loss1, label=label1, linestyle='-',  color='#1f77b4', linewidth=2)  # muted blue
        plt.plot(time2, loss2, label=label2, linestyle='--', color='#ff7f0e', linewidth=2)  # muted orange
        plt.plot(time3, loss3, label=label3, linestyle='-.', color='#2ca02c', linewidth=2)  # muted green
        plt.plot(time4, loss4, label=label4, linestyle=':',  color='#d62728', linewidth=2)  # muted red
                

        plt.xlabel("GPU Time (s)")
        plt.ylabel("Loss")
        plt.title("Loss vs GPU Time")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
# === 训练入口 ===
if __name__ == "__main__":

    # 读取 Excel 文件
    df = pd.read_excel("/home/zhy/Zhou/lm/Ccs/Concrete_Data.xls")

    # 显示列名（确认目标列是哪一个）
    # print(df.columns)

    # 假设最后一列是目标变量（Strength）
    X = df.iloc[:, :-1]  # 所有列（除最后一列）为特征
    y = df.iloc[:, -1]   # 最后一列为目标

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
    print(y_scaled)

    
    # X_tensor = torch.tensor(X.values, dtype=torch.float64)
    # y_tensor = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float64)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float64)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float64)
    print(X_tensor.shape)

    # 训练 LM
    optnum=11100
    model_lm = PINN_LM(X_tensor, y_tensor)
    model_lm.LM(opt_num=optnum, step=1000,bd_tol=0)

    # 训练 MLM
    model_slm = PINN_LM(X_tensor, y_tensor)
    model_slm.SLM(opt_num=optnum, step=1050,bd_tol=0,zone=2/3)
    
    
    model_slmb = PINN_LM(X_tensor, y_tensor)
    model_slmb.SLM(opt_num=optnum, step=1000,bd_tol=1,zone=2/3)
    
    model_lmb = PINN_LM(X_tensor, y_tensor)
    model_lmb.LM(opt_num=optnum, step=1000,bd_tol=1)
    
    # 绘图对比
    plot_loss_vs_time(
        model_lm.loss_record, model_lm.time_record,
        model_slm.loss_record, model_slm.time_record,
        model_lmb.loss_record, model_lmb.time_record,
        model_slmb.loss_record, model_slmb.time_record,
    )
    
    
    
   