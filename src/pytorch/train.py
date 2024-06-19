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
from load import createPMDatasets, createPreloadedPMDatasets, createPMDataloaders
from downsize import MedPool2D
from losses import construct_loss
from utils import get_str_date_time, print_logs, write_hp, val_empty

data_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/data/"
training_dir  = os.path.join(data_dir   , "Training/")
dataset_dir   = os.path.join(data_dir   , "dataset/")
preloaded_dir = os.path.join(dataset_dir, "preloaded/")


def epoch_train(device, model, train_dataloader, loss, optimizer):
    train_loss_sum, nb_batch_done = 0, 0
    for _, batch in enumerate(train_dataloader):
        # Inference
        pics, masks, disk_masks = batch['pic'], batch['mask'], batch['disk_mask']
        pics, masks, disk_masks = pics.to(device), masks.to(device), disk_masks.to(device)
        outputs = torch.squeeze(model(pics), 1) # channel dim is removed
        # Loss
        loss_value = loss(outputs, masks, disk_masks)
        train_loss_sum += loss_value
        nb_batch_done += 1
        # Backpropagation
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
    train_loss = train_loss_sum / nb_batch_done # incomplete batches don't cause pb ('mean')
    return train_loss


def epoch_val(device, model, val_dataloader, loss):
    model.eval()
    val_loss_sum, nb_batch_done = 0, 0
    for _, batch in enumerate(val_dataloader):
        # Inference
        with torch.no_grad():
            pics, masks, disk_masks = batch['pic'], batch['mask'], batch['disk_mask']
            pics, masks, disk_masks = pics.to(device), masks.to(device), disk_masks.to(device)
            outputs = torch.squeeze(model(pics), 1) # channel dim is removed
        # Loss
        loss_value = loss(outputs, masks, disk_masks)
        val_loss_sum += loss_value
        nb_batch_done += 1
    val_loss = val_loss_sum / nb_batch_done # incomplete batches don't cause pb ('mean')
    return val_loss


def train(device, hp, root_dir=None):

    # Create save_dir
    save_dir = os.path.join(training_dir, get_str_date_time())
    os.makedirs(save_dir) if not os.path.exists(save_dir) else print(f"save_dir {save_dir} already exists")
    write_hp(save_dir, hp)

    # Create model
    model = LightModel(alpha_leaky=hp['alpha_leaky'])
    model.init_weights()
    model.to(device)

    # Load data
    rqm_dir = os.path.join(preloaded_dir, f"{hp['resolution'][0]}x{hp['resolution'][1]}_qm{int(hp['qm'])}")
    if os.path.exists(rqm_dir):
        datasets = createPreloadedPMDatasets(rqm_dir)
        root_dir = rqm_dir
    elif not(root_dir is None):
        datasets = createPMDatasets(root_dir, hp['id_pprad_list'], MedPool2D(hp['pool_kernel']))
    else:
        print("Resolution and qm do not match any preloaded data and root_dir was not provided.")
        sys.exit(1)
    dataloaders = createPMDataloaders(datasets, hp['batch_sizes'])

    # Learning objects
    opt = torch.optim.Adam(model.parameters(), lr=hp['initial_lr'])
    lr_scheduler = ReduceLROnPlateau(opt, factor=hp['lr_factor'], patience=hp['lr_patience'], min_lr=hp['lr_min'])
    loss = construct_loss(hp['loss'])

    ##################################
    #       LEARNING PROCESS
    ##################################
    
    if val_empty(root_dir):
        print(f"No val dir: Training for {hp['epochs']} epochs")

        for epoch in range(hp['epochs']):
            # Train
            train_loss = epoch_train(device, model, dataloaders['train'], loss, opt)
            # Learning supervision
            torch.save(model.state_dict(), os.path.join(save_dir, f"model{epoch}.pt"))
            print_logs(epoch, train_loss)
            lr_scheduler.step(train_loss)

    else:
        print(f"Val dir detected: Training with stop_patience={hp['stop_patience']}")
        
        model_path = os.path.join(save_dir, "model.pt")
        best_val_loss, epoch, stop_wait_time = 1e6, 0, 0
        while (epoch < hp['epochs']) and (stop_wait_time <= hp['stop_patience']):
            # Train and val
            train_loss = epoch_train(device, model, dataloaders['train'], loss, opt)
            val_loss   = epoch_val(  device, model, dataloaders['val'  ], loss)
            # Learning supervision
            if (epoch == 0) or (val_loss < best_val_loss - hp['min_delta']):
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_path)
                stop_wait_time = 0
            else:
                stop_wait_time += 1
            print_logs(epoch, train_loss, val_loss=val_loss, stop_wait_time=stop_wait_time)
            lr_scheduler.step(val_loss)
            epoch += 1
