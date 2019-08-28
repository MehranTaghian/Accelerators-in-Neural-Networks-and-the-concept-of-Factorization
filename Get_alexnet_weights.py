import numpy as np
import torch
from AlexNet import AlexNet


# net_data = np.load(open(r"C:\Users\Mehran\Desktop\Lotfi-Kamran\Weights\bvlc_alexnet.npy", "rb"), encoding="latin1",
#                    allow_pickle=True).item()

def get_weights(DIR = 'alex_net'):
    model = AlexNet(10)
    model.load_state_dict(torch.load(DIR))
    weights = {}
    bias = {}
    b = 1
    w = 1
    layers = [l for l in model.parameters()]
    for l in range(len(layers)):
        if l == 2 or l == 5:
            continue
        print(layers[l].data.shape)
        if len(layers[l].data.shape) == 4:
            weights[f'conv{w}'] = layers[l].data.detach().numpy()
            w += 1
        elif len(layers[l].data.shape) == 2:
            weights[f'fc{w}'] = layers[l].data.detach().numpy()
            w += 1
        elif len(layers[l].data.shape) == 1:
            bias[f'b{b}'] = layers[l].data.detach().numpy()
            b += 1
    return weights, bias

# w, b = get_weights()
# s = w['conv5']
# print(s[0].size)