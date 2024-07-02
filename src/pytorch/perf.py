import os
import sys
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ai')))
from light_model import LightModel
from load import get_dataloader, get_items_cat
from losses import construct_loss
from metrics import confusion_mat_rates, acc, precision, recall, F1_score
from utils import get_str_date_time, get_model_path, torch_to_np

data_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data'))
dataset_dir   = os.path.join(data_dir    , "dataset/")
pics_dir      = os.path.join(dataset_dir , "pics/")
masks_dir     = os.path.join(dataset_dir , "masks/")
training_dir  = os.path.join(data_dir    , "Training/")
preloaded_dir = os.path.join(dataset_dir , "preloaded/")


def get_outputs(device, model, str_date_time, subdir):
    dataloader = get_dataloader(str_date_time, subdir)
    model.eval()
    outputs_list = []
    for _, batch in enumerate(dataloader):
        with torch.no_grad():
            pics, masks, disk_masks = batch['pic'], batch['mask'], batch['disk_mask']
            pics, masks, disk_masks = pics.to(device), masks.to(device), disk_masks.to(device)
            outputs_list.append(torch.squeeze(model(pics), 1)) # channel dim is removed
    outputs_tensor = torch.cat(outputs_list)
    return outputs_tensor


def write_perf(device, str_date_time, epoch, subdir):

    #################################
    #           LOADING
    #################################

    # Load data
    with open(os.path.join(training_dir, str_date_time, "hp.json"), 'r') as f:
        hp = json.load(f)
    pics_tensor, masks_tensor, disk_masks_tensor = get_items_cat(str_date_time, subdir)
    masks_arr, disk_masks_arr = masks_tensor.numpy(), disk_masks_tensor.numpy()
    
    # Load model
    model = LightModel(hp['alpha_leaky'])
    model.load_state_dict(torch.load(get_model_path(str_date_time, epoch=epoch)))
    
    # Get preds
    model.eval()
    outputs_tensor = get_outputs(device, model, str_date_time, subdir)
    #preds_tensor = torch.round(torch.sigmoid(outputs_tensor))
    preds_tensor = torch.sigmoid(outputs_tensor)
    preds_arr = torch_to_np(preds_tensor, batched=1)

    #'''
    model_name = os.path.basename(get_model_path(str_date_time, epoch=epoch))[:-3]
    np.save(os.path.join(os.path.join(training_dir, str_date_time), f"{model_name}_preds_{subdir}.npy"), preds_arr)
    rqm_dir = os.path.join(preloaded_dir, f"{hp['resolution'][0]}x{hp['resolution'][1]}_qm{int(hp['qm'])}")
    fn_list = os.listdir(os.path.join(rqm_dir, "pics", subdir))
    fn_list.sort()
    for i in range(preds_arr.shape[0]):
        output = preds_arr[i]
        plt.imshow(output)
        plt.title(fn_list[i])
        plt.colorbar()
        plt.show()
    #'''

    #################################
    #           TEST
    #################################

    test = {}

    # Compute loss
    loss_tensor = construct_loss(hp['loss'])(outputs_tensor, masks_tensor, disk_masks_tensor)
    test['loss'] = float(torch_to_np(loss_tensor))

    # Compute metrics
    nb_ims = pics_tensor.size(0) # int 20
    test_arr = {'TP'  : np.zeros(nb_ims),
                'TN'  : np.zeros(nb_ims),
                'FP'  : np.zeros(nb_ims),
                'FN'  : np.zeros(nb_ims),
                'accu': np.zeros(nb_ims),
                'prec': np.zeros(nb_ims),
                'rec' : np.zeros(nb_ims),
                'f1'  : np.zeros(nb_ims)}
    for i in range(nb_ims):
        cm   = confusion_mat_rates(preds_arr[i], masks_arr[i], disk_masks_arr[i])
        accu = acc(                preds_arr[i], masks_arr[i], disk_masks_arr[i])
        prec = precision(          preds_arr[i], masks_arr[i], disk_masks_arr[i])
        rec  = recall(             preds_arr[i], masks_arr[i], disk_masks_arr[i])
        f1   = F1_score(           preds_arr[i], masks_arr[i], disk_masks_arr[i])
        test_arr[  'TP'][i] = cm['TP']
        test_arr[  'TN'][i] = cm['TN']
        test_arr[  'FP'][i] = cm['FP']
        test_arr[  'FN'][i] = cm['FN']
        test_arr['accu'][i] = accu
        test_arr['prec'][i] = prec
        test_arr[ 'rec'][i] = rec
        test_arr[  'f1'][i] = f1
    for key in ['TP', 'TN', 'FP', 'FN', 'accu', 'prec', 'rec', 'f1']:
        test[key] = np.mean(test_arr[key])

    #################################
    #           WRITE
    #################################

    # Write in save_dir
    test_json = json.dumps(test)
    model_name = os.path.basename(get_model_path(str_date_time, epoch=epoch))[:-3]
    with open(os.path.join(training_dir, str_date_time, f"{model_name}_perf_{subdir}.json"), 'w') as f:
        f.write(test_json)
