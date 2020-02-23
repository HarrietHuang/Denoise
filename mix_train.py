from __future__ import print_function
import argparse
import os
from math import log10
import IPython.display as display
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision import  transforms as T
import torch as t
from color_networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate,simplelayer, my_Loss#, get_style_model_and_losses, ContentLoss
from mix_dataset import datasets,get_training_testing_set
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy

# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset',default='color_gan_vgg' , help='facades')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--direction', type=str, default='a2b', help='a2b or b2a')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default='true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--ganloss', default='true', help='add gan loss')
parser.add_argument('--gan_weight', type=int, default=0.4, help='weight of gan loss')
parser.add_argument('--reload_epo_num', type=int, default=0, help='what is the epo num while reloading')
opt = parser.parse_args()

print(opt)


if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_dataloader ,testing_data_loader= get_training_testing_set()


print('===> Building models')
device = torch.device("cuda:0" if opt.cuda else "cpu")
writer = SummaryWriter()
net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic', gpu_id=device)
simplelayer = simplelayer().to(device)
#cnn = models.vgg19(pretrained=True).features.to(device).eval()

test_mean, test_std = torch.tensor([0.5 ,0.5 ,0.5]), torch.tensor([0.5 ,0.5, 0.5])
criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)

#load checkpoint
# g_load_state_path=r'D:\pix2pix-pytorch-master\checkpoint\color_gan_Nlinear\netG_model_epoch_80.pth'
# d_load_state_path=r'D:\pix2pix-pytorch-master\checkpoint\color_gan_Nlinear\netD_model_epoch_80.pth'
#simplelayer_state_path = r'C:\Users\admin\Download\nccu\project\pix2pix-pytorch-master\checkpoint\jupyter_color_unet\checkpoint_100.pt'

# if simplelayer_state_path:
    # # print('===> loading models')
    # # #simplelayer.load_state_dict(t.load(simplelayer_state_path))
     # net_d = t.load(d_load_state_path)
     # net_g = t.load(g_load_state_path)

print(simplelayer)

# for param in simplelayer.parameters():
    # param.requires_grad = False

if not opt.ganloss:   
    optimizer = optim.Adam(simplelayer.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
else:
    optimizer_g = optim.Adam([{'params': net_g.parameters()},{'params': simplelayer.parameters(), 'lr': 1e-3}],lr=opt.lr, betas=(opt.beta1, 0.999))
    #optimizer_g = optim.Adam([{net_g.parameters()},{simplelayer.parameters(), }], lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
thetaS = 1
lossfn = my_Loss(thetaS).to(device) #instanciate loss function


#simplelayer_scheduler = get_scheduler(optimizer_g, opt)
# net_g_scheduler = get_scheduler(optimizer_g, opt)
# net_d_scheduler = get_scheduler(optimizer_d, opt)   

#real A out, real B groundtruth
#fake B generator out ,real A 

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    # train
    total_loss_mse = 0
    total_loss_g = 0
    total_loss_d = 0
    total_loss_s = 0
    #total_content_loss = 0
    
    for index, (blur, groundtruth, filename) in enumerate(train_dataloader):

        simplelayer.train()
        blur = blur.to(device)
        groundtruth = groundtruth.to(device)
        mid, output = simplelayer(blur)
        mseloss1 = criterionMSE(mid, groundtruth)
        mseloss2 = criterionMSE(output, groundtruth)
        mseloss = mseloss1 + mseloss2
        total_loss_mse += mseloss.item()

        if not opt.ganloss:           
            optimizer.zero_grad()
            mseloss.backward()
            optimizer.step()
            print("===> Epoch[{}]({}/{}): MSELoss: {:.4f}".format(
            epoch + opt.reload_epo_num, index, len(train_dataloader),  mseloss.item()))
            writer.add_scalars('pix2pix', {
                                        'mse':total_loss_mse/len(train_dataloader),                                   
                                        }, epoch + opt.reload_epo_num)
            continue
            
        
        # forward
        real_a = output
        real_b = groundtruth
        fake_b = net_g(real_a)

        ######################
        # (1) Update D network
        ######################

        optimizer_d.zero_grad()
        
        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = net_d.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)
        
        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5
        total_loss_d += loss_d.item()
        if epoch==1:
            loss_d.backward(retain_graph=True)
        else:
            loss_d.backward(retain_graph=True)
       
        optimizer_d.step()

        ######################
        # (2) Update G network
        ######################
      
        optimizer_g.zero_grad()
        
        #saturation
        saturation_loss = lossfn(fake_b)
        total_loss_s +=saturation_loss.item()

        
        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

        # Second, G(A) = B
 
        loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb
        loss_g = (loss_g_gan + loss_g_l1)*opt.gan_weight + mseloss * (1 - opt.gan_weight) + saturation_loss * 0.5
        total_loss_g += loss_g.item() * opt.gan_weight - mseloss.item() * (1 - opt.gan_weight)      
        loss_g.backward(retain_graph=True)
        optimizer_g.step()
             #print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                   # style_score.item(), content_score.item()))
        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f} MSELoss: {:.4f} SatuationLoss: {:.4f} ".format(
            epoch + opt.reload_epo_num, index, len(train_dataloader), loss_d.item(), loss_g.item()- mseloss.item(), mseloss.item(), saturation_loss.item()))

        if opt.ganloss: 
            writer.add_scalars('pix2pix', {'loss_d':(total_loss_d/len(train_dataloader)),
                                        'loss_g':total_loss_g/len(train_dataloader), 
                                        'mse':total_loss_mse/len(train_dataloader),
                                        'saturation':total_loss_s/len(train_dataloader),
                                        }, epoch + opt.reload_epo_num)
        else:
            writer.add_scalars('pix2pix', {
                                        'mse':total_loss_mse/len(train_dataloader),                                   
                                        }, epoch + opt.reload_epo_num)
        writer.flush()
    writer.close()
    # update_learning_rate(net_g_scheduler, optimizer_g)
    # update_learning_rate(net_d_scheduler, optimizer_d)
    
    
    # test
    
    avg_psnr = 0
    for blur, groundtruth, filename in testing_data_loader:
        blur, groundtruth = blur.to(device), groundtruth.to(device)
        
        simplelayer.eval()
        if opt.ganloss:
            _,prediction = simplelayer(blur)
        else:
            net_g = net_g.to(device)
            net_g.eval()
            _,prediction = simplelayer(blur)
            prediction = net_g(prediction)

        #checkpoint
        if epoch  %20 == 0:
        #if epoch == 1:
            if not os.path.exists("checkpoint"):
                os.mkdir("checkpoint")
            if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
                os.mkdir(os.path.join("checkpoint", opt.dataset))
            net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, epoch + opt.reload_epo_num)
            net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.dataset, epoch + opt.reload_epo_num)
            simplelayer_out_path = "checkpoint/{}/simplelayer_model_epoch_{}.pth".format(opt.dataset, epoch + opt.reload_epo_num)
            
            torch.save(simplelayer, simplelayer_out_path)
            torch.save(net_g, net_g_model_out_path)
            torch.save(net_d, net_d_model_out_path)
            print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))
            #count=0
            #save image
        if epoch % 3 == 0 :
            prediction = prediction[0].cpu() * test_std.view(3,1,1) + test_mean.view(3,1,1)
            image_list = T.ToPILImage()(prediction.cpu())

            filepath = r'D:\pix2pix-pytorch-master\{}'.format(opt.dataset)
            if not os.path.exists(filepath):
                os.mkdir(filepath,7777)
            image_list.save(filepath + r'\{}_ganloss0_4.png'.format(filename[0].replace('.png','')),'png')


    