import os
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

def normalize(data):
    data = np.nan_to_num(data)
    data_min, data_max = np.expand_dims(np.min(data,axis=1),axis=1), np.expand_dims(np.max(data,axis=1),axis=1)
    data_norm = np.nan_to_num((data - data_min) / (data_max - data_min))

    return data_norm

def preprocess(args):
    # loading the saved dataset
    DIR_TRAIN = f"./datasets/{args.train_miss_rate}"
    DIR_TEST = f"./datasets/{args.test_miss_rate}"
    DIR_RECONST = "./datasets/0"

    train_path = os.path.join(DIR_TRAIN,"train.npy")
    test_path = os.path.join(DIR_TEST, "test.npy")
    
    train_reconst_path = os.path.join(DIR_RECONST,"train.npy")
    test_reconst_path = os.path.join(DIR_RECONST,"test.npy")

    X_train = np.load(train_path)
    X_test = np.load(test_path)
    X_train_reconst = np.load(train_reconst_path)
    X_test_reconst = np.load(test_reconst_path)

    # making binary masks
    train_mask = np.ones(X_train.shape)
    train_mask[np.isnan(X_train)] = 0
    test_mask = np.ones(X_test.shape)
    test_mask[np.isnan(X_test)] = 0

    # normalizing the data
    train_norm = normalize(X_train)
    train_reconst_norm = normalize(X_train_reconst)
    test_norm = normalize(X_test)
    test_reconst_norm = normalize(X_test_reconst)

    # making the Datasets
    dataset_train = TensorDataset(torch.tensor(train_norm,dtype=torch.float32),torch.tensor(train_mask,dtype=torch.float32),torch.tensor(train_reconst_norm,dtype=torch.float32))
    dataset_test = TensorDataset(torch.tensor(test_norm,dtype=torch.float32),torch.tensor(test_mask,dtype=torch.float32),torch.tensor(test_reconst_norm,dtype=torch.float32))
    # preparing dataloaders
    train_loader = DataLoader(dataset_train,args.batch_size,shuffle=True,num_workers=args.num_workers)
    test_loader = DataLoader(dataset_test,args.batch_size,num_workers=args.num_workers)

    data_objects = {
            "dataset_obj": dataset_train, 
            "train_dataloader": train_loader, 
            "test_dataloader": test_loader,
            "input_dim": X_train.shape[-1],
            "n_train_batches": len(train_loader),
            "n_test_batches": len(test_loader),
            "classif_per_tp": False, 
            "n_labels": 1
    } 
	
    return data_objects