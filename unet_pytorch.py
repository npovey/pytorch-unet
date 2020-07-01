
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt

import numpy as np
import tensorflow.compat.v1 as tfc
import tensorflow as tf
from PIL import Image
import os
from glob import glob


def cal_psnr(im1, im2):
    # assert pixel value range is 0-255 and type is uint8
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    maxval = np.amax(im1)
    psnr = 10 * np.log10(maxval ** 2 / mse)
    return psnr

# Needed to get denoised images six at the time
# can't feed all images into a model.
# X_test: sparseview images
# y_test: clean image
def avg_psnr(model, X_vald, y_vald):
    model.eval()
    num_batches = int(len(X_vald)/6)
    # denoised_image = torch.empty(354,1,512, 512, dtype=torch.float)
    denoised_image = torch.randn(len(X_vald),1,512, 512)
    with torch.no_grad():
        for idx in range(num_batches):
            model.zero_grad()
            denoised_image[idx*6:(idx+1)*6,:,:,:]= model(X_vald[idx*6:(idx+1)*6,:,:,:])

    ## find avg psnr
    psnr_sum = 0
    for i in range(len(X_vald)):
        psnr = cal_psnr(y_vald[i,:,:,:].cpu().data.numpy(), denoised_image[i,:,:,:].cpu().data.numpy())
        psnr_sum += psnr
    avg_psnr = psnr_sum / len(X_vald)
    return avg_psnr

# test 
def test(model, ldct_test, ndct_test, device):
    model.eval()
    X_test = torch.from_numpy(ldct_test).view(len(ldct_test), 1, 512, 512)
    y_test = torch.from_numpy(ndct_test).view(len(ndct_test), 1, 512, 512)
    print("X_test.shape: ", X_test.shape)
    print("y_test.shape: ", y_test.shape)
    X_test, y_test = X_test.to(device), y_test.to(device)

    num_batches = int(len(ndct_test)/6)                                                                                                                       
    denoised_image = torch.empty(len(ndct_test),1,512, 512, dtype=torch.float)
    with torch.no_grad():
        for idx in range(num_batches):
            model.zero_grad()
            denoised_image[idx*6:(idx+1)*6,:,:,:]= model(X_test[idx*6:(idx+1)*6,:,:,:])
    psnr_sum = 0
    for i in range(len(X_test)):
        psnr = cal_psnr(y_test[i,:,:,:].cpu().data.numpy(), denoised_image[i,:,:,:].cpu().data.numpy())
        print("image: ",i ,"PSNR: " , psnr)  
        psnr_sum += psnr
    avg_psnr = psnr_sum / len(X_test)
    print("Avg PSNR: ",avg_psnr)
    return avg_psnr
    # save images as .flt files                                                                                                                         
    #save_dir = "/home/npovey/data/pytorch_models/test"
    #rawfiles = [open(os.path.join(save_dir, "test_{num:08d}.flt".format(num=index)), 'wb') for index in range (354)]
    #for index in range(len(ndct_test)):
    #    img = np.asarray(denoised_image[index,:,:,:])
    #    img.tofile(rawfiles[index])



# test                                                                                                                                                                 
def test_save(model, ldct_test, ndct_test, device):
    model.eval()
    X_test = torch.from_numpy(ldct_test).view(len(ldct_test), 1, 512, 512)
    y_test = torch.from_numpy(ndct_test).view(len(ndct_test), 1, 512, 512)
    print("X_test.shape: ", X_test.shape)
    print("y_test.shape: ", y_test.shape)
    X_test, y_test = X_test.to(device), y_test.to(device)

    num_batches = int(len(ndct_test)/6)
    denoised_image = torch.empty(len(ndct_test),1,512, 512, dtype=torch.float)
    with torch.no_grad():
        for idx in range(num_batches):
            model.zero_grad()
            denoised_image[idx*6:(idx+1)*6,:,:,:]= model(X_test[idx*6:(idx+1)*6,:,:,:])
    psnr_sum = 0
    for i in range(len(X_test)):
        psnr = cal_psnr(y_test[i,:,:,:].cpu().data.numpy(), denoised_image[i,:,:,:].cpu().data.numpy())
        print("image: ",i ,"PSNR: " , psnr)
        psnr_sum += psnr
    avg_psnr = psnr_sum / len(X_test)
    print("Avg PSNR: ",avg_psnr)

    # save images as .flt files
    save_dir = "/home/npovey/data/pytorch_models/test"
    rawfiles = [open(os.path.join(save_dir, "test_{num:08d}.flt".format(num=index)), 'wb') for index in range (354)]
    for index in range(len(ndct_test)):
        img = np.asarray(denoised_image[index,:,:,:])
        img.tofile(rawfiles[index])
    return avg_psnr

def save_png_images_0_1_12(model, ldct_test, device):
    model.eval()

    X_test = torch.from_numpy(ldct_test).view(354, 1, 512, 512)
    print(X_test.shape)
    X_test = X_test.to(device)
    denoised_image= model(X_test[0:2,:,:,:])
                                                                                
    a = denoised_image[0].view(512,512).cpu().data.numpy()
    scalef = np.amax(a)
    a = np.clip(255 * a/scalef, 0, 255).astype('uint8')
    result = Image.fromarray((a).astype(np.uint8))
    result.save('pytorch_unet_0.png')
    
    b = denoised_image[1].view(512,512).cpu().data.numpy()
    scalef = np.amax(b)
    b = np.clip(255 * b/scalef, 0, 255).astype('uint8')
    result = Image.fromarray((b).astype(np.uint8))
    result.save('pytorch_unet_1.png')

    denoised_image= model(X_test[8:10,:,:,:])
    b = denoised_image[0].view(512,512).cpu().data.numpy()
    scalef = np.amax(b)
    b = np.clip(255 * b/scalef, 0, 255).astype('uint8')
    result = Image.fromarray((b).astype(np.uint8))
    result.save('pytorch_unet_8.png')

    denoised_image= model(X_test[102:104,:,:,:])
    b = denoised_image[0].view(512,512).cpu().data.numpy()
    scalef = np.amax(b)
    b = np.clip(255 * b/scalef, 0, 255).astype('uint8')
    result = Image.fromarray((b).astype(np.uint8))
    result.save('pytorch_unet_102.png')

    denoised_image= model(X_test[12:14,:,:,:])    
    b = denoised_image[0].view(512,512).cpu().data.numpy()
    scalef = np.amax(b)
    b = np.clip(255 * b/scalef, 0, 255).astype('uint8')
    result = Image.fromarray((b).astype(np.uint8))
    result.save('pytorch_unet_12.png')

    b = X_test[12].view(512,512).cpu().data.numpy()
    scalef = np.amax(b)
    b = np.clip(255 * b/scalef, 0, 255).astype('uint8')
    result = Image.fromarray((b).astype(np.uint8))
    result.save('pytorch_unet_12_ldct.png')


#def denoise_all(model):
#    print("denoising all")
#    model.eval()
#    for i in range(9):
#        print("i: ",i)
#        num_batches = int(400/4)
#        ldct_train7 = ldct_train[(i*400):(i+1)*400,:,:,:]
#        ldct_train7 = ldct_train7.reshape(400,1,512,512)
#        X_train = torch.from_numpy(ldct_train7)
#        X_train = X_train.to(device)
#
#        denoised_image = torch.empty(400,1,512, 512, dtype=torch.float)
#        with torch.no_grad():
#            for idx in range(num_batches):
#                # print("batch_number",idx)
#                model.zero_grad()
#                denoised_image[idx*4:(idx+1)*4,:,:,:]= model(X_train[idx*4:(idx+1)*4,:,:,:])
#                # save images as .flt files
#        save_dir = "/home/npovey/data/pytorch_models/denoised_images"
#        rawfiles = [open(os.path.join(save_dir, "test_{num:08d}.flt".format(num=index+(i*400))), 'wb') for index in range (400)]
#        for index in range(400):
#            # print(index+(i*400))
#            img = np.asarray(denoised_image[index,:,:,:])
#            img.tofile(rawfiles[index])



# Unet model with all filters from UNET orgiginal paper
# Define model
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        # print(1)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(num_features=64)
        self.conv2 =  nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(num_features=64)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        
        # print(2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.batch3 = nn.BatchNorm2d(num_features=128)
        self.conv4 =  nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.batch4 = nn.BatchNorm2d(num_features=128)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        
        # print(3)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.batch5 = nn.BatchNorm2d(num_features=256)
        self.conv6 =  nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.batch6 = nn.BatchNorm2d(num_features=256)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        # print(4)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.batch7 = nn.BatchNorm2d(num_features=512)
        self.conv8 =  nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.batch8 = nn.BatchNorm2d(num_features=512)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        
        # print(5)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.batch9 = nn.BatchNorm2d(num_features=1024)
        self.conv10 =  nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.batch10 = nn.BatchNorm2d(num_features=1024)

        # print(6)
        self.trans1 = nn.ConvTranspose2d(in_channels=1024,out_channels=512, kernel_size=(2, 2), stride=2, padding=0)
        ## concatenate [channels must be add]
        self.conv11 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
        self.batch11 = nn.BatchNorm2d(num_features=512)
        self.conv12 =  nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.batch12 = nn.BatchNorm2d(num_features=512)

        # print(7)
        self.trans2 = nn.ConvTranspose2d(in_channels=512,out_channels=256, kernel_size=(2, 2), stride=2, padding=0)
        ## concatenate [channels must be added]
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.batch13 = nn.BatchNorm2d(num_features=256)
        self.conv14 =  nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.batch14 = nn.BatchNorm2d(num_features=256)
        
        # print(8)
        self.trans3 = nn.ConvTranspose2d(in_channels=256,out_channels=128, kernel_size=(2, 2), stride=2, padding=0)
        ## concatenate [channels must be added]
        self.conv15 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.batch15 = nn.BatchNorm2d(num_features=128)
        self.conv16 =  nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.batch16 = nn.BatchNorm2d(num_features=128)

        # print(9)
        self.trans4 = nn.ConvTranspose2d(in_channels=128,out_channels=64, kernel_size=(2, 2), stride=2, padding=0)
        ## concatenate [channels must be added]
        self.conv17 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.batch17 = nn.BatchNorm2d(num_features=64)
        self.conv18 =  nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.batch18 = nn.BatchNorm2d(num_features=64)
        self.conv19 =  nn.Conv2d(64, out_channels=1, kernel_size=1, padding=0)

    def forward(self, inp):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # print("A")
        x = self.conv1(inp)
        x = self.batch1(x)
        x = F.relu(x)
        c1 = self.conv2(x)
        x = self.batch2(c1)
        x = F.relu(x)
        x = self.pool1(x)

        # print("B")
        x = self.conv3(x)
        x = self.batch3(x)
        x = F.relu(x)
        c2 = self.conv4(x)
        x = self.batch4(c2)
        x = F.relu(x)
        x = self.pool2(x)

        # print("C")
        x = self.conv5(x)
        x = self.batch5(x)
        x = F.relu(x)
        c3 = self.conv6(x)
        x = self.batch6(c3)
        x = F.relu(x)
        x = self.pool3(x)

        # print("D")
        x = self.conv7(x)
        x = self.batch7(x)
        x = F.relu(x)
        c4 = self.conv8(x)
        x = self.batch8(c4)
        x = F.relu(x)
        x = self.pool4(x)

        # print("E")
        x = self.conv9(x)
        x = self.batch9(x)
        x = self.conv10(x)
        x = self.batch10(x)

        # print("F")
        u1 = self.trans1(x)
        x = torch.cat((u1, c4),1)
        x = self.conv11(x)
        x = self.batch11(x)
        x = F.relu(x)
        x = self.conv12(x)
        x = self.batch12(x)
        x = F.relu(x)
        
        # print("G")
        u2 = self.trans2(x)
        x = torch.cat((u2, c3),1)
        x = self.conv13(x)
        x = self.batch13(x)
        x = F.relu(x)
        x = self.conv14(x)
        x = self.batch14(x)
        x = F.relu(x)

        # print("H")
        u3 = self.trans3(x)
        x = torch.cat((u3, c2),1)
        x = self.conv15(x)
        x = self.batch15(x)
        x = F.relu(x)
        x = self.conv16(x)
        x = self.batch16(x)
        x = F.relu(x)

        # print("I")
        u4 = self.trans4(x)
        x = torch.cat((u4, c1),1)
        x = self.conv17(x)
        x = self.batch17(x)
        x = F.relu(x)
        x = self.conv18(x)
        x = self.batch18(x)
        x = F.relu(x)
        x = self.conv19(x)
        x = inp - x
        return x


def train(model,ndct_train, ldct_train, ldct_vald, ndct_vald, device,PATH):
    print("len(ndct_train): ", len(ndct_train))
    print("len(ldct_train): " ,len(ldct_train))
    count = 0
    psnr_set = {0}
    print("training")
    model.train()
    start = time.time()
    n_epochs = 1   # < ---- change to 100
    batch_size = 4
    length = len(ndct_train)
    losses = []
    psnrs = []
    loss_func = nn.MSELoss()
    print('iter,\tloss')

    set_size = 360
    z = int(set_size/batch_size)
    num_sets = int(3240/set_size)
    for epoch in range(n_epochs):
        if(epoch == 0):
            optim = torch.optim.Adam(model.parameters(), lr=0.0001)
        if(epoch < 10 and epoch !=0):
            optim = torch.optim.Adam(model.parameters(), lr=0.001)
            print("lr: 0.001")
        elif(epoch < 20):
            optim = torch.optim.Adam(model.parameters(), lr=0.0005)
            print("lr: 0.0005")
        elif(epoch < 30):
            optim = torch.optim.Adam(model.parameters(), lr=0.00025)
            print("lr: 0.00025")
        elif(epoch < 40):
            optim = torch.optim.Adam(model.parameters(), lr=0.000125)
            print("lr:  0.000125")
        elif(epoch < 50):
            optim = torch.optim.Adam(model.parameters(), lr=0.0000635) 
            print("lr: 0.0000635")
        else:
            optim = torch.optim.Adam(model.parameters(), lr=0.00003175)
            print("lr:  0.00003175")

        print()  
        print("Epoch",epoch)
        for i in range (num_sets):
            print("set of " ,set_size,": ", i)
            ldct_train7 = ldct_train[i*set_size:(i+1)*set_size,:,:,:]
            ndct_train7 = ndct_train[i*set_size:(i+1)*set_size,:,:,:]
            p = np.random.permutation(set_size)
            ldct_train7 = ldct_train7[p,:,:,:]
            ndct_train7 = ndct_train7[p,:,:,:]
            
            ldct_train2 = ldct_train7.reshape(set_size,1,512,512)
            ndct_train2 = ndct_train7.reshape(set_size,1,512,512)
            
            X_torch = torch.from_numpy(ldct_train2)
            y_torch = torch.from_numpy(ndct_train2)
            X_torch, y_torch = X_torch.to(device), y_torch.to(device)

            for i in range(z):
                optim.zero_grad()
                y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
                loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
                losses.append(loss) 
                loss.backward()
                optim.step()
                if i % 100 == 0:
                    print('batch: {},\t{:.7f}'.format(i, loss.item()))
          
            # # # -----------------flipud augment images ------------------                                                                               
            flipped_l = np.flipud(ldct_train7)
            flipped_m = np.flipud(ndct_train7)

            flipped_l = flipped_l.reshape(set_size,1,512,512)
            flipped_m = flipped_m.reshape(set_size,1,512,512)

            X_torch = torch.from_numpy(flipped_l.copy())
            y_torch = torch.from_numpy(flipped_m.copy())
            X_torch, y_torch = X_torch.to(device), y_torch.to(device)

            for i in range(z):
                optim.zero_grad()
                y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
                loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
                losses.append(loss)
                loss.backward()
                optim.step()
                if i % 100 == 0:
                    print('batch: {},\t{:.7f}'.format(i, loss.item()))


            # # # -----------------rotate 90 degrees augment images ------------------                                                                    

            flipped_l = np.rot90(ldct_train7, axes=(-2,-1))
            flipped_m = np.rot90(ndct_train7, axes=(-2,-1))
            
            flipped_l = flipped_l.reshape(set_size,1,512,512)
            flipped_m = flipped_m.reshape(set_size,1,512,512)

            X_torch = torch.from_numpy(flipped_l.copy())
            y_torch = torch.from_numpy(flipped_m.copy())
            X_torch, y_torch = X_torch.to(device), y_torch.to(device)

            for i in range(z):
                optim.zero_grad()
                y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
                loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
                losses.append(loss)
                loss.backward()
                optim.step()
                if i % 100 == 0:
                    print('batch: {},\t{:.7f}'.format(i, loss.item()))


            # -----------------rotate 90+flip degrees augment images ------------------                                                                   
            flipped_l = np.rot90(ldct_train7, axes=(-2,-1))
            flipped_m = np.rot90(ndct_train7, axes=(-2,-1))
            flipped_l = np.flipud( flipped_l)
            flipped_m = np.flipud( flipped_m)
            
            flipped_l = flipped_l.reshape(set_size,1,512,512)
            flipped_m = flipped_m.reshape(set_size,1,512,512)

            X_torch = torch.from_numpy(flipped_l.copy())
            y_torch = torch.from_numpy(flipped_m.copy())
            X_torch, y_torch = X_torch.to(device), y_torch.to(device)

            for i in range(z):
                optim.zero_grad()
                y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
                loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
                losses.append(loss)
                loss.backward()
                optim.step()
                if i % 100 == 0:
                    print('batch: {},\t{:.7f}'.format(i, loss.item()))

            # -----------------rotate 180 degrees augment images ------------------                      
            flipped_l = np.rot90(ldct_train7, k=2, axes=(-2,-1))
            flipped_m = np.rot90(ndct_train7, k=2, axes=(-2,-1))
            
            flipped_l = flipped_l.reshape(set_size,1,512,512)
            flipped_m = flipped_m.reshape(set_size,1,512,512)

            X_torch = torch.from_numpy(flipped_l.copy())
            y_torch = torch.from_numpy(flipped_m.copy())
            X_torch, y_torch = X_torch.to(device), y_torch.to(device)

            for i in range(z):
                optim.zero_grad()
                y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
                loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
                losses.append(loss)
                loss.backward()
                optim.step()
                if i % 100 == 0:
                    print('batch: {},\t{:.7f}'.format(i, loss.item()))

            # -----------------rotate 180 + flip degrees augment images ------------------                                                           
            flipped_l = np.rot90(ldct_train7, k=2, axes=(-2,-1))
            flipped_m = np.rot90(ndct_train7, k=2, axes=(-2,-1))
            flipped_l = np.flipud( flipped_l)
            flipped_m = np.flipud( flipped_m)
            
            flipped_l = flipped_l.reshape(set_size,1,512,512)
            flipped_m = flipped_m.reshape(set_size,1,512,512)

            X_torch = torch.from_numpy(flipped_l.copy())
            y_torch = torch.from_numpy(flipped_m.copy())
            X_torch, y_torch = X_torch.to(device), y_torch.to(device)

            for i in range(z):
                optim.zero_grad()
                y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
                loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
                losses.append(loss)
                loss.backward()
                optim.step()
                if i % 100 == 0:
                    print('batch: {},\t{:.7f}'.format(i, loss.item()))


            # -----------------rotate 270 degrees augment images ------------------                                                          
            flipped_l = np.rot90(ldct_train7,k=3, axes=(-2,-1))
            flipped_m = np.rot90(ndct_train7,k=3, axes=(-2,-1))
            

            flipped_l = flipped_l.reshape(set_size,1,512,512)
            flipped_m = flipped_m.reshape(set_size,1,512,512)

            X_torch = torch.from_numpy(flipped_l.copy())
            y_torch = torch.from_numpy(flipped_m.copy())
            X_torch, y_torch = X_torch.to(device), y_torch.to(device)

            for i in range(z):
                optim.zero_grad()
                y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
                loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
                losses.append(loss)
                loss.backward()
                optim.step()
                if i % 100 == 0:
                    print('batch: {},\t{:.7f}'.format(i, loss.item()))


            # -----------------rotate 270 + flip degrees augment images ------------------                                                           
            flipped_l = np.rot90(ldct_train7,k=3, axes=(-2,-1))
            flipped_m = np.rot90(ndct_train7,k=3, axes=(-2,-1))
            flipped_l = np.flipud( flipped_l)
            flipped_m = np.flipud( flipped_m)

            flipped_l = flipped_l.reshape(set_size,1,512,512)
            flipped_m = flipped_m.reshape(set_size,1,512,512)

            X_torch = torch.from_numpy(flipped_l.copy())
            y_torch = torch.from_numpy(flipped_m.copy())
            X_torch, y_torch = X_torch.to(device), y_torch.to(device)

            for i in range(z):
                optim.zero_grad()
                y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
                loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
                losses.append(loss)
                loss.backward()
                optim.step()
                if i % 100 == 0:
                    print('batch: {},\t{:.7f}'.format(i, loss.item()))
            #-------------- end augmented  -------------                                                                                                               

        curr_psnr = test(model, ldct_vald, ndct_vald, device)
        print('epoch: {},\tAvg PSNR {:.7f}'.format(epoch, curr_psnr))
        
        model.train() 
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss,
        }, PATH)    
        
        # stop training if model didn't improve for 6 epochs
        print("count: ", count)
        if(curr_psnr <= max(psnr_set)):
            count = count + 1
            print("count increased by one:...........", count)
        else:
            print("new  max............... ", curr_psnr)
            count = 0
        if(count == 10):
            end = time.time()
            print("Total training time: ", end - start)
            print("Done training")
            return 0
        psnr_set.add(curr_psnr)
        print("psnr_set: ", psnr_set)
        psnrs.append(curr_psnr)

    psnr = test(model, ldct_vald, ndct_vald,device)
    print('epoch: {},\tAvg PSNR {:.7f}'.format(epoch, psnr))
    torch.cuda.empty_cache()
    np.save('losses', losses) 
    np.save('psnrs', psnrs)    
    end = time.time()
    print("Total training time: ", end - start)


def main():
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")


    #ldct_train = np.load('/home/npovey/data/new_idea/sparseview_60_train_3600.npy') # loads saved array into variable sparseview_60_train.
    ndct_train = np.load('/home/npovey/data/new_idea/ndct_train_3600.npy') # loads saved array into variable ndct_train.
    #ldct_test = np.load('/home/npovey/data/new_idea/sparseview_60_test_354.npy') # loads saved array into variable sparseview_60_test.
    ndct_test = np.load('/home/npovey/data/new_idea/ndct_test_354.npy') # loads saved array into variable ndct_test.
    
    # -- load sparseview_90 data ------#                
    
    #ldct_train_strs = sorted(glob('/data/CT_data/sparseview/sparseview_90/train/*.flt'))
    #ldct_train_strs = sorted(glob('/data/CT_data/sparseview/sparseview_180/train/*.flt'))
    #ldct_train_strs = sorted(glob('/data/CT_data/images/ldct_7e4/train/*.flt'))
    #ldct_train_strs = sorted(glob('/data/CT_data/images/ldct_1e5/train/*.flt'))
    #ldct_train_strs = sorted(glob('/data/CT_data/images/ldct_2e5/train/*.flt'))


    #print("len(ldct_train_strs): ", len(ldct_train_strs))
    
    #ldct_train = []
    #for i in range(0, len(ldct_train_strs)):
    #    f = open(ldct_train_strs[i],'rb')
    #    a = np.fromfile(f, np.float32)
    #    ldct_train.append(a)
    #    f.close()
    #print("len(ldct_train)....: ",len(ldct_train))
    #ldct_train2 = np.asarray(ldct_train)
    #ldct_train2 = ldct_train2.reshape(3600,512,512,1)
    # np.save('ldct_1e5_train_3600', ldct_train2)
    #np.save('ldct_2e5_train_3600', ldct_train2)

    #ldct_train = np.load('/home/npovey/data/pytorch_models/ldct_7e4_train_3600.npy')
    #ldct_train = np.load('/home/npovey/data/pytorch_models/ldct_1e5_train_3600.npy')
    #ldct_train = np.load('/home/npovey/data/pytorch_models/ldct_2e5_train_3600.npy')

    
    #ldct_test_strs = sorted(glob('/data/CT_data/sparseview/sparseview_90/test/*.flt'))
    #ldct_test_strs = sorted(glob('/data/CT_data/sparseview/sparseview_180/test/*.flt'))
    #ldct_test_strs = sorted(glob('/data/CT_data/images/ldct_7e4/test/*.flt'))                                                                                                       
    # ldct_test_strs = sorted(glob('/data/CT_data/images/ldct_1e5/test/*.flt'))
    #ldct_test_strs = sorted(glob('/data/CT_data/images/ldct_2e5/test/*.flt'))

    #print("len(ldct_test_strs) ", len(ldct_test_strs))    
    #ldct_test = []
    #for i in range(0, len(ldct_test_strs)):
    #    f = open(ldct_test_strs[i],'rb')
    #    a = np.fromfile(f, np.float32)
    #    ldct_test.append(a)
    #    f.close()
    #print("len(ldct_test)....: ",len(ldct_test))
    #ldct_test2 = np.asarray(ldct_test)
    #ldct_test2 = ldct_test2.reshape(354,512,512,1)
    #np.save('ldct_7e4_test_354.npy', ldct_test2)  
    #np.save('ldct_1e5_test_354.npy', ldct_test2)
    #np.save('ldct_2e5_test_354.npy', ldct_test2)

    #ldct_test = np.load('/home/npovey/data/pytorch_models/sparseview_180_test_354.npy')
    #ldct_test = np.load('/home/npovey/data/pytorch_models/ldct_7e4_test_354.npy')
    #ldct_test = np.load('/home/npovey/data/pytorch_models/ldct_1e5_test_354.npy')
    #ldct_test = np.load('/home/npovey/data/pytorch_models/ldct_2e5_test_354.npy')

    # ------end sparsview 90 data load ----
    ldct_train = np.load('/home/npovey/data/pytorch_models/sparseview_90_train_3600.npy')
    ldct_test = np.load('/home/npovey/data/pytorch_models/sparseview_90_test_354.npy')

    #ldct_train = np.load('/home/npovey/data/pytorch_models/sparseview_180_train_3600.npy')
    #ldct_test = np.load('/home/npovey/data/pytorch_models/sparseview_180_test_354.npy')

    #ldct_train = np.load('/home/npovey/data/pytorch_models/ldct_7e4_train_3600.npy')
    #ldct_test = np.load('/home/npovey/data/pytorch_models/ldct_7e4_test_354.npy')

    vald_start = 3240
    vald_end = 3600
    vald_len = 360

    ldct_vald = ldct_train[vald_start:vald_end, :, :, :]
    ndct_vald = ndct_train[vald_start:vald_end, :, :, :]
    ldct_vald = ldct_vald.reshape(vald_len, 1, 512, 512)
    ndct_vald = ndct_vald.reshape(vald_len, 1, 512, 512)


    ldct_train = ldct_train[0:3240, :, :, :]
    ndct_train = ndct_train[0:3240, :, :, :]
    ldct_train = ldct_train.reshape(3240, 1, 512, 512)
    ndct_train = ndct_train.reshape(3240, 1, 512, 512)

    ldct_test = ldct_test.reshape(354, 1, 512, 512)
    ndct_test = ndct_test.reshape(354, 1, 512, 512)
    
    PATH = "unet_weights.pth"
    model = Unet()
#    model = DnCNN()
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)


    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("pytorch_total_params: ", pytorch_total_params)

    train(model, ndct_train, ldct_train, ldct_vald, ndct_vald, device, PATH)
    torch.cuda.empty_cache()

    test_save(model, ldct_test, ndct_test,device)
    save_png_images_0_1_12(model, ldct_test,device)
    torch.cuda.empty_cache()
    print("done training")

if __name__ == "__main__":
    main()


