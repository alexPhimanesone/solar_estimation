import os
import sys
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
    'C:/Users/aphimaneso/Work/Projects/mmsegmentation/src/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
    'C:/Users/aphimaneso/Work/Projects/mmsegmentation/src/ai/')))
from light_model import LightModel
from load import createPMDatasets, createPMDataloaders
from downsize import MedPool2D
from utils import get_str_date_time, print_logs, write_hp, str_to_loss, val_empty

data_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/data/"
training_dir  = os.path.join(data_dir   , "Training/")


def epoch_train(device, model, train_dataloader, loss, optimizer):
    train_loss_sum, nb_batch_done = 0, 0
    for _, batch in enumerate(train_dataloader):
        # Inference
        pics, masks = batch['pic'], batch['mask']
        pics, masks = pics.to(device), masks.to(device)
        outputs = model(pics)
        # Loss
        loss_value = loss(outputs, masks)
        train_loss_sum += loss_value
        nb_batch_done += 1
        # Backpropagation
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
    train_loss = train_loss_sum / nb_batch_done
    return train_loss


def epoch_val(device, model, val_dataloader, loss):
    model.eval()
    val_loss_sum, nb_batch_done = 0, 0
    for _, batch in enumerate(val_dataloader):
        # Inference
        with torch.no_grad():
            pics, masks = batch['pic'], batch['mask']
            pics, masks = pics.to(device), masks.to(device)
            outputs = model(pics)
        # Loss
        loss_value = loss(outputs, masks)
        val_loss_sum += loss_value
        nb_batch_done += 1
    val_loss = val_loss_sum / nb_batch_done
    return val_loss


def train(device, root_dir, hp):

    # Create save_dir
    save_dir = os.path.join(training_dir, get_str_date_time())
    os.makedirs(save_dir) if not os.path.exists(save_dir) else print(f"save_dir {save_dir} already exists")
    write_hp(save_dir, hp)

    # Create model
    model = LightModel(alpha_leaky=hp['alpha_leaky'])
    model.init_weights()
    model.to(device)
    model_path = os.path.join(save_dir, "model.pt")

    # Load data
    datasets = createPMDatasets(root_dir, hp['id_pprad_list'], MedPool2D(hp['pool_kernel']))
    dataloaders = createPMDataloaders(datasets, hp['batch_sizes'])

    # Learning objects
    opt = torch.optim.Adam(model.parameters(), lr=hp['initial_lr'])
    lr_scheduler = ReduceLROnPlateau(opt, factor=hp['lr_factor'], patience=hp['lr_patience'], min_lr=hp['lr_min'])
    loss = str_to_loss(hp['loss'])

    ##################################
    #       LEARNING PROCESS
    ##################################
    
    if val_empty(root_dir):

        for epoch in range(hp['epochs']):
            # Train
            train_loss = epoch_train(device, model, dataloaders['train'], loss, opt)
            # Learning supervision
            torch.save(model, model_path)
            print_logs(epoch, train_loss)
            lr_scheduler.step(train_loss)

    else:

        best_val_loss, epoch, stop_wait_time = 1e6, 0, 0
        while (epoch < hp['epochs']) and (stop_wait_time <= hp['stop_patience']):
            # Train and val
            train_loss = epoch_train(device, model, dataloaders['train'], loss, opt)
            val_loss   = epoch_val(  device, model, dataloaders['val'  ], loss)
            # Learning supervision
            if (epoch == 0) or (val_loss < best_val_loss - hp['min_delta']):
                best_val_loss = val_loss
                torch.save(model, model_path)
                stop_wait_time = 0
            else:
                stop_wait_time += 1
            print_logs(epoch, train_loss, val_loss=val_loss, stop_wait_time=stop_wait_time)
            lr_scheduler.step(val_loss)
            epoch += 1
        