'''
purpose: tiling the WSI and make the prediction using parallel process

author: Hongming Xu
email: mxu@ualberta.ca

modified by : Sumanth Reddy Nakkireddy
email:nakkireddy.sumanthreddy@mayo.edu
'''
import time
import torch
import openslide
import matplotlib.pyplot as plt
import concurrent.futures
# import numpy as np
import cupy as np
from itertools import repeat
from PIL import Image
from skimage import transform
import torchvision.transforms.functional as F
from MacenkoNormalizer import MacenkoNormalizer
#import czifile as czi   # intall it on lab server???

def wsi_tiling_pred(File,dest_imagePath,img_name,Tile_size,model,class_ind,device,thr=0.5, parallel_running=True):
    '''
    thr: threshold to be applied on prediction maps
    parallel_running=False -> for debug by developers
    '''
    
    since = time.time()
    # open image
    try :
        Slide = openslide.OpenSlide(File)
    except :
        print('openslide error!')
        return None

    xr = float(Slide.properties['openslide.mpp-x'])  # pixel resolution at x direction
    yr = float(Slide.properties['openslide.mpp-y'])  # pixel resolution at y direction
    # generate X, Y coordinates for tiling
    Stride = [round(Tile_size[0] / xr), round(Tile_size[1] / yr)]
    Dims = Slide.level_dimensions
    X = np.arange(0, Dims[0][0] + 1, Stride[0])
    Y = np.arange(0, Dims[0][1] + 1, Stride[1])
    X, Y = np.meshgrid(X, Y)

    pred_c=np.zeros((X.shape[0]-1,X.shape[1]-1,3),'uint8')
    pred_g=np.zeros((X.shape[0]-1,X.shape[1]-1,3),'float')

    global pred_c_g
    pred_c_g = pred_c
    global pred_g_g
    pred_g_g = pred_g
    global thr_g
    thr_g=thr

    global grid_g # save top-left coordinates of interested tiles
    grid_g=[]
    if parallel_running==True:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
             for _ in executor.map(parallel_filling, list(range(X.shape[0]-1)), repeat(X), repeat(Y),repeat(Stride),repeat(File),repeat(model),repeat(class_ind),repeat(device)):
                 pass

        img1 = Image.fromarray(pred_c_g.get())
        img2 = Image.fromarray(pred_g_g.get().astype('uint8'))
        img1.save(dest_imagePath + img_name.split('.')[0] + '_' + 'color.png')
        img2.save(dest_imagePath + img_name.split('.')[0] + '_' + 'gray.png')
    else: # sequential running for debugging
        for i in range(X.shape[0] - 1):
            for j in range(X.shape[1] - 1):
                    Tile = Slide.read_region((int(X[i, j]), int(Y[i, j])), 0, (Stride[0], Stride[1]))
                    Tile = np.asarray(Tile)
                    Tile = Tile[:, :, :3]
                    bn=np.sum(Tile[:, :, 0] < 5) + np.sum(np.mean(Tile,axis=2) > 245)
                    if (np.std(Tile[:,:,0])+np.std(Tile[:,:,1])+np.std(Tile[:,:,2]))/3>18 and bn<Stride[0]*Stride[1]*0.3:
                        #Tile=transform.resize(Tile,[224,224], order=1, preserve_range=True)
                        Tile = Image.fromarray(Tile)
                        Tile=F.resize(Tile,[224,224])
                        inputs=np.asarray(Tile).transpose(2,0,1)
                        inputs/=255.0
                        inputs = torch.from_numpy(inputs).float()
                        inputs.unsqueeze_(0)
                        inputs=inputs.to(device)
                        with torch.no_grad():
                            outputs=model(inputs)
                        pred=outputs.cpu().numpy()
                        pred_g_g[i, j, :] = pred[0,class_ind] * 255

                        if pred[0,class_ind]>thr:
                            pred_c_g[i,j,0]=255
                            grid_g.append((int(X[i, j]), int(Y[i, j])))
                        else:
                            pred_c_g[i,j,2]=255


    time_elapsed = time.time() - since
    print('Mapping complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    return grid_g, Stride

def parallel_filling(i,X,Y,Stride,File,model,class_ind,device):
    Slide = openslide.OpenSlide(File)

    for j in range(X.shape[1] - 1):
        Tile = Slide.read_region((int(X[i, j]), int(Y[i, j])), 0, (Stride[0], Stride[1]))
        Tile = np.asarray(Tile)
        Tile = Tile[:, :, :3]
        bn = np.sum(Tile[:, :, 0] < 5) + np.sum(np.mean(Tile,axis=2) > 240) # previously 245
        if (np.std(Tile[:, :, 0]) + np.std(Tile[:, :, 1]) + np.std(Tile[:, :, 2])) / 3 > 18 and bn < Stride[0] * Stride[1] * 0.3:
            #Tile = transform.resize(Tile, [224, 224], order=1, preserve_range=True)
            Tile = Image.fromarray(Tile.get())
            Tile = F.resize(Tile, [224, 224])
            inputs = np.asarray(Tile).transpose(2, 0, 1)
            inputs = inputs/ 255.0
            inputs = torch.from_numpy(inputs.get()).float()
            inputs.unsqueeze_(0)
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(inputs)
            pred = outputs.cpu().numpy()
            pred_g_g[i, j, :] = pred[0,class_ind] * 255

            if pred[0,class_ind] > thr_g:
                pred_c_g[i, j, 0] = 255
                grid_g.append((int(X[i, j]), int(Y[i, j])))
            else:
                pred_c_g[i, j, 2] = 255

#------------------second version-------------------#
def wsi_tiling_pred_ii(File,dest_imagePath,img_name,Tile_size,model,class_ind,device,
                       model1,output_path1,class_ind1,parallel_running=True):
    '''
    In this version:
    two models are loaded to make predictions, e.g., tumor prediction -> msi prediction
    parallel_running=False -> for debug by developers
    '''
    
    since = time.time()
    # open image
    try :
        Slide = openslide.OpenSlide(File)
    except :
        print('openslide error!')
        return -1

    xr = float(Slide.properties['openslide.mpp-x'])  # pixel resolution at x direction
    yr = float(Slide.properties['openslide.mpp-y'])  # pixel resolution at y direction
    # generate X, Y coordinates for tiling
    Stride = [round(Tile_size[0] / xr), round(Tile_size[1] / yr)]
    Dims = Slide.level_dimensions
    X = np.arange(0, Dims[0][0] + 1, Stride[0])
    Y = np.arange(0, Dims[0][1] + 1, Stride[1])
    X, Y = np.meshgrid(X, Y)

    pred0=np.zeros((X.shape[0]-1,X.shape[1]-1,3),'float')
    pred1=np.zeros((X.shape[0]-1,X.shape[1]-1,3),'float')
    pred_ind=np.zeros((X.shape[0]-1,X.shape[1]-1),'float')

    global pred0_g
    pred0_g = pred0
    global pred1_g
    pred1_g = pred1
    global pred_ind_g
    pred_ind_g=pred_ind

    global model1_g
    model1_g=model1
    global class_ind1_g
    class_ind1_g=class_ind1

    reference_path = './macenko_reference_img.png'
    try:
        # Initialize the Macenko normalizer
        reference_img = np.array(
            Image.open(reference_path).convert('RGB'))
        normalizer = MacenkoNormalizer()
        normalizer.fit(reference_img)

        global normalizer_g
        normalizer_g = normalizer
    except:
        print('no given reference image for color normalization~~~~~')

    if parallel_running==True:
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
             for _ in executor.map(parallel_filling_ii, list(range(X.shape[0]-1)), repeat(X), repeat(Y),repeat(Stride),repeat(File),repeat(model),repeat(class_ind),repeat(device)):
                 pass

    else:  # sequential running for debugging
        for i in range(X.shape[0] - 1):
            for j in range(X.shape[1] - 1):
                Tile = Slide.read_region((int(X[i, j]), int(Y[i, j])), 0, (Stride[0], Stride[1]))
                Tile = np.asarray(Tile)
                Tile = Tile[:, :, :3]
                bn = np.sum(Tile[:, :, 0] < 5) + np.sum(np.mean(Tile, axis=2) > 245)
                if (np.std(Tile[:, :, 0]) + np.std(Tile[:, :, 1]) + np.std(Tile[:, :, 2])) / 3 > 18 and bn < Stride[0] * \
                        Stride[1] * 0.3:
                    # Tile=transform.resize(Tile,[224,224], order=1, preserve_range=True)
                    try:
                        Tile = normalizer_g.transform(Tile)
                    except:
                        print('i=%d,j=%d' % (i, j))
                        continue

                    Tile = Image.fromarray(Tile)
                    Tile = F.resize(Tile, [224, 224])
                    inputs = np.asarray(Tile).transpose(2, 0, 1)
                    inputs = inputs / 255.0
                    inputs = torch.from_numpy(inputs).float()
                    inputs.unsqueeze_(0)
                    inputs = inputs.to(device)
                    #inputs = inputs.cuda()
                    with torch.no_grad():
                        outputs = model(inputs)
                    pred = outputs.cpu().numpy()
                    pred0_g[i, j, :] = pred[0, class_ind] * 255

                    if pred[0, class_ind] > 0.5:  # 0.5 could be adaptively changed
                        with torch.no_grad():
                            outputs1= model1_g(inputs)
                        pred1 = outputs1.cpu().numpy()
                        pred1_g[i, j, :] = pred1[0, class_ind1_g] * 255
                        pred_ind_g[i, j] = 1

    img1 = Image.fromarray(pred0_g.get().astype('uint8'))
    img1.save(dest_imagePath + img_name.split('.')[0] + '_' + 'gray.png')

    img2 = Image.fromarray(pred1_g.get().astype('uint8'))
    img2.save(output_path1 + img_name.split('.')[0] + '_' + 'gray.png')

    ch1=pred1_g[:,:,0]/255.0
    pred=np.mean(ch1[pred_ind_g==1])
    time_elapsed = time.time() - since
    print('Mapping complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return pred

def parallel_filling_ii(i,X,Y,Stride,File,model,class_ind,device):
    Slide = openslide.OpenSlide(File)

    for j in range(X.shape[1] - 1):
        Tile = Slide.read_region((int(X[i, j]), int(Y[i, j])), 0, (Stride[0], Stride[1]))
        Tile = np.asarray(Tile)
        Tile = Tile[:, :, :3]
        bn = np.sum(Tile[:, :, 0] < 5) + np.sum(np.mean(Tile,axis=2) > 245)
        if (np.std(Tile[:, :, 0]) + np.std(Tile[:, :, 1]) + np.std(Tile[:, :, 2])) / 3 > 18 and bn < Stride[0] * Stride[1] * 0.3:
            #Tile = transform.resize(Tile, [224, 224], order=1, preserve_range=True)
            try:
                Tile = normalizer_g.transform(Tile)
            except:
                print('i=%d,j=%d' % (i, j))
                continue

            Tile = Image.fromarray(Tile.get())
            Tile = F.resize(Tile, [224, 224])
            inputs = np.asarray(Tile).transpose(2, 0, 1)
            inputs = inputs/255.0
            inputs = torch.from_numpy(inputs.get()).float()
            inputs.unsqueeze_(0)
            inputs = inputs.to(device)
            #inputs = inputs.cuda()
            with torch.no_grad():
                outputs = model(inputs)
            pred = outputs.cpu().numpy()
            pred0_g[i, j, :] = pred[0,class_ind] * 255

            if pred[0,class_ind] > 0.5:     # 0.5 could be adaptively changed
                with torch.no_grad():
                    outputs1 = model1_g(inputs)
                pred1=outputs1.cpu().numpy()
                pred1_g[i, j, :] = pred1[0,class_ind1_g] * 255
                pred_ind_g[i,j]=1

##---------tiling and prediction for .czi slides-------------##
def parallel_filling_czi(i,X,Y,Stride,model,class_ind,device):
    for j in range(X.shape[1] - 1):
        Tile = c_wsi[int(X[i, j]):int(X[i, j] + Stride[0]), int(Y[i, j]):int(Y[i, j] + Stride[1]), :]
        bn = np.sum(Tile[:, :, 0] < 5) + np.sum(np.mean(Tile, axis=2) > 210)
        if (np.std(Tile[:, :, 0]) + np.std(Tile[:, :, 1]) + np.std(Tile[:, :, 2])) / 3 > 18 and bn < Stride[0] * Stride[
            1] * 0.3:
            Tile = Image.fromarray(Tile)
            Tile = F.resize(Tile, [224, 224])
            inputs = np.asarray(Tile).transpose(2, 0, 1)
            inputs = inputs / 255.0
            inputs = torch.from_numpy(inputs).float()
            inputs.unsqueeze_(0)
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(inputs)
            pred = outputs.cpu().numpy()
            pred_g_g[i, j, :] = pred[0, class_ind] * 255

            if pred[0, class_ind] > thr_g:
                pred_c_g[i, j, 0] = 255
            else:
                pred_c_g[i, j, 2] = 255

def wsi_tiling_pred_czi(File,dest_imagePath,img_name,Tile_size,model,class_ind,device,thr = 0.5,parallel_running = True):

    since = time.time()
    # open image
    czi_obj = czi.CziFile(File)

    # num_tiles = len(czi_obj.filtered_subblock_directory)
    # tile_dims_dict =czi_obj.metadata(raw=False)['ImageDocument']['Metadata']['Experiment']['ExperimentBlocks']['AcquisitionBlock'][
    #    'SubDimensionSetups']['RegionsSetup']['SampleHolder']['TileDimension']

    global c_wsi
    c_wsi = np.zeros(czi_obj.shape[2:], np.uint8)
    for i, directory_entry in enumerate(czi_obj.filtered_subblock_directory):
        subblock = directory_entry.data_segment()
        tile = subblock.data(resize=False, order=0)
        xs = directory_entry.start[2] - czi_obj.start[2]
        xe = xs + tile.shape[2]
        ys = directory_entry.start[3] - czi_obj.start[3]
        ye = ys + tile.shape[3]

        c_wsi[xs:xe, ys:ye, :] = tile.squeeze()

    xr = czi_obj.metadata(raw=False)['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][0]['Value'] * 1e+6
    yr = czi_obj.metadata(raw=False)['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][1]['Value'] * 1e+6
    # generate X, Y coordinates for tiling
    Stride = [round(Tile_size[0] / xr), round(Tile_size[1] / yr)]
    Dims = c_wsi.shape
    X = np.arange(0, Dims[0] + 1, Stride[0])
    Y = np.arange(0, Dims[1] + 1, Stride[1])
    X, Y = np.meshgrid(X, Y)

    pred_c=np.zeros((X.shape[0]-1,X.shape[1]-1,3),'uint8')
    pred_g=np.zeros((X.shape[0]-1,X.shape[1]-1,3),'float')

    global thr_g
    thr_g=thr
    global pred_c_g
    pred_c_g = pred_c
    global pred_g_g
    pred_g_g = pred_g
    if parallel_running==True:
        # parallel-running
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
             for _ in executor.map(parallel_filling_czi, list(range(X.shape[0]-1)), repeat(X), repeat(Y),repeat(Stride),repeat(model),repeat(class_ind),repeat(device)):
                 pass

        img1 = Image.fromarray(pred_c_g)
        img2 = Image.fromarray(pred_g_g.astype('uint8'))
        img1.save(dest_imagePath + img_name.split('.')[0] + '_' + 'color.png')
        img2.save(dest_imagePath + img_name.split('.')[0] + '_' + 'gray.png')
    else: # sequential running for debugging
        for i in range(X.shape[0] - 1):
            for j in range(X.shape[1] - 1):
                Tile = c_wsi[int(X[i, j]):int(X[i, j] + Stride[0]), int(Y[i, j]):int(Y[i, j] + Stride[1]), :]
                bn = np.sum(Tile[:, :, 0] < 5) + np.sum(np.mean(Tile, axis=2) > 210)
                if (np.std(Tile[:, :, 0]) + np.std(Tile[:, :, 1]) + np.std(Tile[:, :, 2])) / 3 > 18 and bn < Stride[0] * \
                        Stride[1] * 0.3:
                    Tile = Image.fromarray(Tile)
                    Tile = F.resize(Tile, [224, 224])
                    inputs = np.asarray(Tile).transpose(2, 0, 1)
                    inputs = inputs / 255.0
                    inputs = torch.from_numpy(inputs).float()
                    inputs.unsqueeze_(0)
                    inputs = inputs.to(device)
                    with torch.no_grad():
                        outputs = model(inputs)
                    pred = outputs.cpu().numpy()
                    pred_g_g[i, j, :] = pred[0, class_ind] * 255

                    if pred[0, class_ind] > thr_g:
                        pred_c_g[i, j, 0] = 255
                    else:
                        pred_c_g[i, j, 2] = 255

    time_elapsed = time.time() - since
    print('Mapping complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
