import os
from os.path import join as opj
import sys
import numpy as np
import cv2
import mmcv
from mmengine import Config
from mmseg.apis import MMSegInferencer, init_model, inference_model, show_result_pyplot
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ai')))
from metrics import confusion_mat
from navig_dataset import get_id_endroit
from utils import get_last_model_path, squeeze_mask

data_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data'))
cropped_dir = opj(data_dir, "dataset", "mmseg_orga", "cropped")
training_dir = opj(data_dir, "Training")

train_timestamp = "0702-1952"
iter = 60
config_fn = "unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py"
set = "test"


def get_cm_arr():
    
    # Initiate model
    config_path = opj(training_dir, train_timestamp, config_fn)
    if iter is None:
        checkpoint_path = get_last_model_path(train_timestamp)
    else:
        checkpoint_path = opj(training_dir, train_timestamp, f"iter_{iter}.pth")
    model = init_model(config_path, checkpoint_path, 'cpu')

    # Run inference
    id_pprad_arr = np.load(opj(cropped_dir, "id_pprad_arr.npy"))
    print(id_pprad_arr.shape)
    print(type(id_pprad_arr[0]))
    img_set = opj(cropped_dir, "img_dir", set)
    list_img_fn = os.listdir(img_set)
    cm_arr = np.zeros((len(list_img_fn), 4))
    for i in range(len(list_img_fn)):
        img_fn = list_img_fn[i]
        img_path = opj(img_set, img_fn)
        result = inference_model(model, img_path)
        pred_data = result.pred_sem_seg.data[0].numpy()
        gt_data = squeeze_mask(mmcv.imread(opj(cropped_dir, "ann_dir", set, f"{img_fn[:7]}.png")))
        print(pred_data.shape)
        print(gt_data.shape)
        print(np.unique(pred_data))
        print(np.unique(gt_data))
        disk_mask = id_pprad_arr[int(get_id_endroit(id_pic=img_fn[:7]))]
        print(disk_mask.shape)
        print(np.unique(disk_mask))
        cm = confusion_mat(pred_data, gt_data, disk_mask)
        cm_arr[i] = cm['TP'], cm['TN'], cm['FP'], cm['FN']
        print(img_fn)
    
    return cm_arr


def compute_metrics(cm_arr):

    # Sum TP, FP, TN, FN across all images
    TP_total = np.sum(cm_arr[0])
    TN_total = np.sum(cm_arr[1])
    FP_total = np.sum(cm_arr[2])
    FN_total = np.sum(cm_arr[3])

    precision = TP_total / (TP_total + FP_total) if (TP_total + FP_total) != 0 else 0 # Precision
    recall = TP_total / (TP_total + FN_total) if (TP_total + FN_total) != 0 else 0 # Recall
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0 #  F1 Score
    global_accuracy = (TP_total + TN_total) / (TP_total + TN_total + FP_total + FN_total) # Global Accuracy
    return precision, recall, f1_score, global_accuracy


def see_preds():
    
    # Initiate model
    config_path = opj(training_dir, train_timestamp, config_fn)
    if iter is None:
        checkpoint_path = get_last_model_path(train_timestamp)
    else:
        checkpoint_path = opj(training_dir, train_timestamp, f"iter_{iter}.pth")
    model = init_model(config_path, checkpoint_path, 'cpu')

    # Run inference
    img_set = opj(data_dir, "dataset", "mmseg_orga", "cropped", "img_dir", set)
    list_img_fn = os.listdir(img_set)
    for i in range(len(list_img_fn)):
        img_fn = list_img_fn[i]
        img_path = opj(img_set, img_fn)
        result = inference_model(model, img_path)
        show_result_pyplot(model, img_path, result, out_file=opj(training_dir, train_timestamp, set, img_fn), show=False)
        print(img_fn)


def inspect_dims():
    from mmengine.dataset import Compose
    import torch
    
    # Initiate model
    config_path = opj(training_dir, train_timestamp, config_fn)
    checkpoint_path = get_last_model_path(train_timestamp)
    model = init_model(config_path, checkpoint_path, 'cpu')

    # Choose an image (random)
    img_original_path = opj(data_dir, "dataset", "mmseg_orga", "img_dir", "test", "0008002.jpg")
    img_original = mmcv.imread(img_original_path)
    
    # Preprocess image
    cfg = model.cfg
    for t in cfg.test_pipeline:
        if t.get('type') == 'LoadAnnotations':
            cfg.test_pipeline.remove(t)
    pipeline = Compose(model.cfg.test_pipeline)
    data_ = dict(img_path=img_original_path)
    data_ = pipeline(data_)
    #print(data_['inputs'].shape) # 384, 512
    img_preprocessed = data_['inputs']

    # Write preprocessed image
    img_preprocessed_np = img_preprocessed.detach().numpy().transpose(1, 2, 0)
    img_preprocessed_path = opj(training_dir, train_timestamp, "img_preprocessed.jpg")
    cv2.imwrite(img_preprocessed_path, img_preprocessed_np)
    
    # Inference
    result = inference_model(model, img_original_path)
    #result = inference_model(model, img_preprocessed_path)
    output_sem_seg = result.pred_sem_seg

    # Visualize
    show_result_pyplot(model, img_original_path, result, out_file=opj(training_dir, train_timestamp, "temp", "test.jpg"))

    print("SHAPES")
    print(img_original.shape)
    print(img_preprocessed.shape)
    print(output_sem_seg.shape)


'''
import numpy as np
import matplotlib.pyplot as plt
from crop_around_disk import crop_around_disk, get_disk_mask
from downsize_mask import min_pooling
from metrics import confusion_mat, acc, plot_cm

resolution = (512, 512)
id_pprad = 0 # le même pprad que celui sélectionné pour l'inférence
pprad_path = os.path.join(pprads_dir, f"pprad{id_pprad}.yml")

# load pred
pred = np.load(os.path.join(data_dir, "pred.npy"))
pred = np.where(pred == 10, 255, 0)

# load gt mask
height, width = get_height_width(id_pic)
mask = read_raw_image(os.path.join(masks_dir, f"mask{id_pic}.raw"), width, height)
mask = crop_around_disk(pprad_path, mask)
disk_mask = get_disk_mask(pprad_path)
mask, disk_mask_downsized = min_pooling(mask, resolution, disk_mask)

# Metrics
cm = confusion_mat(pred, mask, disk_mask_downsized)
accu = acc(pred, mask, disk_mask_downsized)

# Plots
nb_tot = cm['TP'] + cm['TN'] + cm['FP'] + cm['FN']
for key in cm.keys():
    print(f'{key}: {cm[key] / nb_tot}')
print(f"acc: {accu}")
plot_cm(pred, mask, disk_mask_downsized)
'''
