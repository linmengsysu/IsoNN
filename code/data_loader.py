import numpy as np
import pickle
import torch
from torchvision import datasets
from torchvision import transforms
# from torch.utils.data.sampler import SubsetRandomSampler


def load_dataset(data_dir, fold_count):
    print('load_dataset')
    filename = data_dir + '/fold_' + fold_count
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    train_y = data['train']['y'].reshape(-1)
    # print('train_y', train_y)
    train_y[train_y<0] = 0
    # train_y[train_y>0] = 1
    # train_y = train_y.squeeze(-1)
    index = np.arange(len(train_y))
    # np.random.shuffle(index)
    train_graph = data['train']['X'][index]
    train_y = train_y[index]
    train_label = np.zeros((len(data['train']['X']), 2))


    test_y = data['test']['y'].reshape(-1)
    test_y[test_y<0] = 0
    index = np.arange(len(test_y))
    
    test_y = test_y[index]
    test_graph = data['test']['X']
    

    (n_graph, hw) = train_graph.shape
    n_H = int(np.sqrt(float(hw)))
    test_graph = np.array(test_graph)


    train_graph = train_graph.reshape(n_graph, 1, n_H, n_H)
    test_graph = test_graph.reshape(-1, 1, n_H, n_H)

    train_data = []
    test_data = []

    for i in range(len(train_graph)):
        train_data.append((torch.from_numpy(train_graph[i]).float(), torch.from_numpy(np.array(train_y[i])).long()))
    for i in range(len(test_graph)):
        test_data.append((torch.from_numpy(test_graph[i]).float(), torch.from_numpy(np.array(test_y[i])).long()))
    return train_data, test_data

 

def get_train_loader(train_data, batch_size, random_seed, shuffle=True):
  

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
  
    return train_loader


def get_test_loader(test_data, batch_size):
  
    data_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
    )

    return data_loader


