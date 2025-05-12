import torch
import torch
import torch.nn as nn
import collections
import numpy as np
# %matplotlib widget
from matplotlib import pyplot as plt
from numpy import matrix as mat
from torch.nn.modules import loss
from torch import  vmap
from torch.func import jacrev ,functional_call

torch.cuda.init()

act=torch.nn.Tanh()

# model = nn.Sequential(
#             nn.Linear(in_features=100, out_features=5),
#             act,
#             nn.Linear(in_features=5, out_features=1, bias=False)
#         )
input_size=2
hidden_size1=10
hidden_size2=10
output_size=1
model=nn.Sequential(
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
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
model.to(device)

batch_size = 10000
input_features = 2
input_data = torch.randn(batch_size, input_features, device=device)  #init input             
input_data.requires_grad=True
func_params=dict(model.named_parameters())

# print(func_params)         
def fm(params,input):
    return functional_call(model,params,input).squeeze(0)

# loss function                   
def floss(func_params,input):      
    output=fm(func_params,input)
    du_d1 = torch.autograd.grad(
                        inputs=input,
                        outputs=output,
                        grad_outputs=torch.ones((batch_size,1)).to(device),
                        retain_graph=True,
                        create_graph=True
                        )[0]
                    
    du_d1.unsqueeze(-1)
    # print(du_d1[:,[0]].shape)
    du_dxx = torch.autograd.grad(
        inputs=input,
        outputs=du_d1[:,[0]],
        grad_outputs=torch.ones((batch_size,1)).to(device),
        retain_graph=True,
        create_graph=True
        )[0]
    du_dyy = torch.autograd.grad(
        inputs=input,
        outputs=du_d1[:,[1]],
        grad_outputs=torch.ones((batch_size,1)).to(device),
        retain_graph=True,
        create_graph=True
        )[0]
    fx = du_dyy+du_dxx + 68*torch.sin(8*input[0])*torch.sin(2*input[1])
    print('sb',fx.shape)
    return fx



params_list = list(func_params.values())
print(len(params_list))
random_matrix = torch.randint(0, 2, (hidden_size1, hidden_size2)).to(device)
print('list',params_list[2].shape)
c=params_list[2]
# c=func_params['layer1.weight']
# print(len(c.view(-1)))
basis_vector=torch.eye(batch_size).to(device)
basis_vector=basis_vector
# print(basis_vector.shape)
# loss=floss(func_params,input_data)
def grad(inputs,v):
    loss=floss(func_params,input_data)
    return torch.autograd.grad(loss,c,torch.ones_like(loss),retain_graph=True)

# def grad(inputs,v):
#     loss=floss(func_params,input_data)
#     return torch.autograd.grad(loss,c,torch.ones_like(loss),retain_graph=True)

gradient=vmap(grad,(None,0))(input_data,basis_vector)
print('gradient on c',gradient)
per_sample_gradient=vmap(grad,(None,0))(input_data,basis_vector)
# print('per_sample_grad',per_sample_gradient.shape)

cnt=0
for g in gradient: 
    g = g.detach()
    J_d = g.reshape(len(g),-1) if cnt == 0 else torch.hstack([J_d,g.reshape(len(g),-1)])
    cnt = 1
print('Jacobi:',J_d.shape)
# start_event2 = torch.cuda.Event(enable_timing=True)
# end_event2 = torch.cuda.Event(enable_timing=True)
# start_event2.record()

# per_sample_grads =vmap(jacrev(floss,0), (None, 0))(func_params, input_data)
# # g=vmap(grad)(basis_vector)


# end_event2.record()
# torch.cuda.synchronize()  # Wait for the events to be recorded!
# elapsed_time_ms = start_event2.elapsed_time(end_event2)
# print(elapsed_time_ms)



# print(per_sample_grads)
# for k,g in per_sample_grads.items(): 
#     print(g)
#i want to get the gradient of ['0.weight']
# oweightgrads=vmap(grad,(0,None))(floss(0,input_data),func_params['0.weight'])
# gradient=[grad(v.unsqueeze(-1)) for v in basis_vector.unbind()]
# print('grad:',gradient)




    








            
