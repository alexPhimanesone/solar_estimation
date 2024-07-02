import os
from os.path import join as opj
import sys
import json
import matplotlib.pyplot as plt
sys.path.append(os.path.normpath(opj(os.path.dirname(os.path.abspath(__file__)), '..')))
from utils import get_vis_path

data_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data'))
training_dir = opj(data_dir, "Training")


def vis_train(timestamp):

    # Load logs file
    vis_path = get_vis_path(timestamp)
    vis_data = []
    with open(vis_path) as f:
        for line in f:
            if "lr" in line: # ensure that it's a train step log
                log = json.loads(line)
                vis_data.append(log)
    
    # Parse logs
    train_iter, train_lr, train_loss = [], [], []
    for log in vis_data:
        train_iter.append(log["iter"])
        train_lr.append(log["lr"])
        train_loss.append(log["loss"])

    # Plotting
    plt.figure()
    plt.plot(train_iter, train_lr, label='train lr', linestyle='--')
    plt.title('Train LR Over Iterations')
    plt.xlabel('Iter')
    plt.ylabel('LR')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.figure()
    plt.plot(train_iter, train_loss, label='train loss', linestyle='solid')
    plt.title('Train Loss Over Iterations')
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def vis_val(timestamp):

    # Load logs file
    vis_path = get_vis_path(timestamp)
    vis_data = []
    with open(vis_path) as f:
        for line in f:
            if not "lr" in line: # ensure that it's not a train step log
                log = json.loads(line)
                vis_data.append(log)

    # Parse logs
    val_steps, val_aAcc, val_mIoU, val_mAcc = [], [], [], []
    for log in vis_data:
        val_steps.append(log["step"])
        val_aAcc.append(log["aAcc"])
        val_mIoU.append(log["mIoU"])
        val_mAcc.append(log["mAcc"])

    # Plotting
    plt.figure()
    plt.plot(val_steps, val_aAcc, label='Val aAcc', linestyle='--')
    plt.plot(val_steps, val_mIoU, label='Val mIoU', linestyle='solid')
    plt.plot(val_steps, val_mAcc, label='Val mAcc', linestyle=':')
    plt.xlabel('Step')
    plt.title('Validation metrics Over Steps')
    plt.legend()
    plt.grid(True)
    plt.show()

