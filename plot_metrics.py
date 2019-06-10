import numpy as np
import torch
import matplotlib.pyplot as plt


plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率

def plot_comp_val_ade(my_checkpoint, original_checkpoint, name, num_len):

    my_train_ade = my_checkpoint['metrics_val_epoch']['ade'][:num_len]
    original_train_ade = original_checkpoint['metrics_val_epoch']['ade'][:num_len]
    x_axix = [i for i in range(len(original_train_ade))]
    sub_axix = x_axix

    plt.title(str(my_checkpoint['args']['dataset_name']).upper())
    plt.plot(sub_axix, my_train_ade, color='green', label='My Model')
    plt.plot(sub_axix, original_train_ade, color='red', label='Original Model')
    plt.legend() 
    plt.xlabel('epochs')
    plt.ylabel('ADE')
    plt.savefig('images/'+str(my_checkpoint['args']['dataset_name']).upper()+'_'+ \
                str(my_checkpoint['args']['pred_len']).upper() +'_ADE_COMP_' + str(name))
    plt.show()
    
def plot_comp_val_fde(my_checkpoint, original_checkpoint, name, num_len):
    
    my_train_ade = my_checkpoint['metrics_val_epoch']['fde'][:num_len]
    original_train_ade = original_checkpoint['metrics_val_epoch']['fde'][:num_len]

    x_axix = [i for i in range(len(original_train_ade))]

    sub_axix = x_axix

    plt.title(str(my_checkpoint['args']['dataset_name']).upper())
    plt.plot(sub_axix, my_train_ade, color='green', label='My Model')
    plt.plot(sub_axix, original_train_ade, color='red', label='Original Model')
    plt.legend() # 显示图例
    plt.xlabel('epochs')
    plt.ylabel('FDE')
    plt.savefig('images/'+str(my_checkpoint['args']['dataset_name']).upper()+'_'+ \
                str(my_checkpoint['args']['pred_len']).upper() +'_FDE_COMP_' + str(name))
    plt.show()
    
    
def plot_train_val_fde(checkpoint):

    train_ade = checkpoint['metrics_train_epoch']['fde']
    val_ade = checkpoint['metrics_val_epoch']['fde']
    x_axix = [i for i in range(len(train_ade))]
    sub_axix = x_axix

    plt.title(str(checkpoint['args']['dataset_name']).upper())
    plt.plot(x_axix, train_ade, color='green', label='train')
    plt.plot(sub_axix, val_ade, color='red', label='val')
    
    plt.ylim((0.5, 3.5))
    my_y_ticks = np.arange(0.5, 3.5, 0.5)
    plt.yticks(my_y_ticks)
    
    plt.legend() # 显示图例

    plt.xlabel('epochs')
    plt.ylabel('FDE')
    plt.savefig('images/'+str(checkpoint['args']['dataset_name']).upper()+'_'+ \
                str(checkpoint['args']['pred_len']).upper() +'_FDE_TRAIN_VAL')
    plt.show()
    
def plot_train_val_ade(checkpoint):

    train_ade = checkpoint['metrics_train_epoch']['ade']
    val_ade = checkpoint['metrics_val_epoch']['ade']
    x_axix = [i for i in range(len(train_ade))]
    sub_axix = x_axix

    plt.title(str(checkpoint['args']['dataset_name']).upper())
    plt.plot(x_axix, train_ade, color='green', label='train')
    plt.plot(sub_axix, val_ade, color='red', label='val')
    
    
    plt.ylim((1.0,3))
    my_y_ticks = np.arange(1, 3.2, 0.5)
    plt.yticks(my_y_ticks)
    
    plt.legend() # 显示图例

    plt.xlabel('epochs')
    plt.ylabel('ADE')
    plt.savefig('images/'+str(checkpoint['args']['dataset_name']).upper()+'_'+ \
                str(checkpoint['args']['pred_len']).upper() +'_ADE_TRAIN_VAL')
    plt.show()