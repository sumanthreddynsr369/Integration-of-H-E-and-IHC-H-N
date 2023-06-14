'''
purpose: loead pretrained deep learning models to make predictions on wsis
e.g., tils, or msi prediction

author: Hongming Xu
email:mxu@ualberta.ca

modified by : Sumanth Reddy Nakkireddy
email:nakkireddy.sumanthreddy@mayo.edu
'''
import os
from torchvision import models
import torch.nn as nn
from wsi_tiling_pred import *
from datetime import datetime

import argparse

gpu = "cuda:1"

wsi_ext = '.ndpi'

def freeze_weights(module):
    for param in module.parameters():
        param.requires_grad = False

def recursively_enumerate_model(module):
    if list(module.children()) == []:
        return [module]
    else:
        enumerated_model = []
        for child in module.children():
            enumerated_model += recursively_enumerate_model(child)
        return enumerated_model

def build_model(n_classes, fp):
    model_ft = models.resnet18(pretrained=True)

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, n_classes)

    layers = recursively_enumerate_model(model_ft)
    layers = [layer for layer in layers if (type(layer) != nn.BatchNorm2d and len(list(layer.parameters())) > 0)]
    for layer in layers[:round(fp * len(layers))]:
        freeze_weights(layer)

    return model_ft


def test_end_to_end(class_num, model_name, fp, op, lr, batch_size, model_dir,
                    class_interest, tile_size, wsi_path, output_path):
    '''
    end_to_end testing for easy usages
    input: path to wsi
    output: prediction masks saved in output path
    '''

    # s1: load model
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    model = build_model(class_num, fp)
    model.to(device)


    model.load_state_dict(torch.load(model_dir + "{}_{}_{}_{}_{}.pt".format(
        model_name, fp, op, lr, batch_size)))
    model = nn.Sequential(model, nn.Softmax(dim=1))
    model.eval()

    # s2:
    class_ind = class_interest
    wsis = sorted(os.listdir(wsi_path))
    thr = 0.5  # 0.4->tils, 0.5->tumor

    for img_name in wsis[:]:
        if wsi_ext in img_name:
            pid = img_name.split('.')[0]
            print("patient id is %s" % pid)
            
            if os.path.exists(output_path + img_name.split('.')[0] + '_' + 'color.png') and os.path.exists(output_path + img_name.split('.')[0] + '_' + 'gray.png') :
                print("This slide is already done!!")
                continue
                
                
            if wsi_ext == '.czi':
                wsi_tiling_pred_czi(wsi_path + '/' + img_name, output_path, img_name, tile_size, model,
                                    class_ind, device, thr)
            else:
                wsi_tiling_pred(wsi_path + '/' + img_name, output_path, img_name, tile_size,
                                model, class_ind, device, thr)


def test_end_to_end_ii(tile_size, wsi_path, output_path):
    '''
    end_to_end testing for easy usages
    input: path to wsi
    output: prediction masks saved in output path

    In this version, we assume that two models are sequentially applied on the wsi
    e.g., wsi -> tumor detector -> msi prediction
    '''

    # s1: load tumor detection model
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    class_name = ['adimuc', 'strmus', 'tumstu']  # see data fold names
    class_ind0 = 2
    model0 = build_model(len(class_name), 0)
    model0.to(device)

    model0.load_state_dict(torch.load('./tumor_models/coad_read/' + "{}_{}_{}_{}_{}.pt".format(
        'resnet18', 0, 'adam', 0.0001, 4)))#, map_location='cpu'))
    model0 = nn.Sequential(model0, nn.Softmax(dim=1))
    model0.eval()
    output_path_tumor = output_path + '/pred_tumor/'

    # s2: load msi_prediction model
    class_name1 = ['msi_h', 'mss']
    class_num1 = len(class_name1)
    class_ind1 = 0
    model1 = build_model(class_num1, 0)
    model1.to(device)
    model1.load_state_dict(torch.load('./msi_models/coad_read/' + "{}_{}_{}_{}_{}.pt".format(
        'resnet18', 0, 'adam', 0.0001, 4)))#, map_location='cpu'))
    model1 = nn.Sequential(model1, nn.Softmax(dim=1))
    model1.eval()
    output_path_msi = output_path + '/pred_msi/'

    wsis = sorted(os.listdir(wsi_path))
    wsis_name = []
    wsis_pred = []
    wsis_gt = []
    for img_name in wsis[:]:
        if wsi_ext in img_name:
            pid = img_name.split('.')[0]
            starttime = datetime.now()
            print("patient id is %s" % pid)
            
            if os.path.exists(output_path_tumor + img_name.split('.')[0] + '_' + 'gray.png') and os.path.exists(output_path_msi + img_name.split('.')[0] + '_' + 'gray.png') :
                print("This slide is already done!!")
                continue
                                
            pred = wsi_tiling_pred_ii(wsi_path + '/' + img_name, output_path_tumor, img_name, tile_size, model0, class_ind0, device,
                                      model1, output_path_msi, class_ind1)
            print('msi-h prediction probability=%f' % pred)
            print(datetime.now()-starttime)

            wsis_name.append(pid)
            wsis_pred.append(pred)

    return wsis_name, wsis_pred, wsis_gt

def get_args():
    '''Parses args. Must include all hyperparameters you want to tune.'''

    parser = argparse.ArgumentParser()
    parser.add_argument(
          '-a','--analysis',
          required=True,
          type=str,          
          help='m: msi, s: stromal, t: til') 
    parser.add_argument(
          '-i','--input',
          required=True,
          type=str,          
          help='path of input wsi data')
    parser.add_argument(
          '-o','--output',
          required=True,
          type=str,          
          help='path for output')
    parser.add_argument(
          '-g','--gpu',          
          nargs="?",
          type=int,
          default=0,
          const='-1',
          help='Input GPU Number you want to use.')
                           
    args = parser.parse_args()
    return args
                           
if __name__ == '__main__':
    args = get_args()    

    if args.gpu > -1 :
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,"+str(args.gpu)  # The GPU id to use, usually either "0" or "1";        
        
    
    if not os.path.exists(args.output) :
        os.makedirs(args.output)
    
    if args.analysis == 'm' :
        if not os.path.exists(args.output + '/pred_tumor') :
            os.makedirs(args.output + '/pred_tumor')
            
        if not os.path.exists(args.output + '/pred_msi') :
            os.makedirs(args.output + '/pred_msi')
            
        tile_size = [256, 256]
        test_end_to_end_ii(tile_size, args.input, args.output)
    elif args.analysis == 's' :
        if not os.path.exists(args.output + '/pred_stromal') :
            os.makedirs(args.output + '/pred_stromal')
                        
        class_name = ['others', 'stromal']  # see data fold names
        class_interest = 1
        model_dir = './stromal_models/pan_cancer/'
        tile_size = [112, 112]
        output_path = args.output + '/pred_stromal/'
        test_end_to_end(len(class_name), 'resnet18', 0, 'adam', 0.001, 64, model_dir,
                        class_interest, tile_size, args.input, output_path)
    elif args.analysis == 't' :
        if not os.path.exists(args.output + '/pred_tils') :
            os.makedirs(args.output + '/pred_tils')
                        
        class_name = ['others', 'tils']  # see data fold names
        class_interest = 1
        model_dir = './til_models/pan_cancer/'
        tile_size = [112, 112]
        output_path = args.output + '/pred_tils/'
        test_end_to_end(len(class_name), 'resnet18', 0, 'adam', 0.0001, 4, model_dir,
                        class_interest, tile_size, args.input, output_path)        
    else:
        raise RuntimeError('Undefined selecton!!!!')
