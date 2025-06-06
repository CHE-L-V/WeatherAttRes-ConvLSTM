# src/utils.py
import numpy as np
import torch

def normalize(data, mean, std):
    return (data - mean) / std

def denormalize(data, mean, std):
    return data * std + mean

def to_tensor(data):
    return torch.from_numpy(data).float()

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model
