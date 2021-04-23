import torch

pretrained = torch.load('../output/pretrain/extractor/iteration_500000.pth')

for k in pretrained.keys():
    print(k.split('.', 1))

    i, o = k.split('.', 1)
    
    key0 = '0' + '.' + o
    key1 = '1' + '.' + o
    key2 = '2' + '.' + o
    pretrained[key1] = pretrained[key0]
    pretrained[key2] = pretrained[key0]

    if int(i) == str:
        break

torch.save(pretrained, 'edited.pth')