import argparse, os, sys
sys.path.insert(0, '..')
proj_root = os.path.join('..')

import itertools
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from srdense.cyclenet import Generator as Generator
from srdense.cyclenet import Discriminator as Discriminator

from srdense.cyclenet import  L1_Charbonnier_loss
from srdense.ssim import SSIM

from srdense.dataset import DatasetGAN as DataSet
from srdense.proj_utils.plot_utils import *
from srdense.utils import *


model_name = 'cycle_super'
model_folder = os.path.join(proj_root, 'model_adam', model_name) 
if not os.path.exists(model_folder):
    os.mkdir(model_folder)

super_loss_plot = plot_scalar(name = "super_loss",  env= model_name, rate = 1000)
loss_G_identity_plot = plot_scalar(name = "loss_G_identity",  env= model_name, rate = 1000)
loss_G_GAN_plot = plot_scalar(name = "loss_G_GAN",  env= model_name, rate = 1000)
loss_G_cycle_plot = plot_scalar(name = "loss_G_cycle",  env= model_name, rate = 1000)
loss_D_plot = plot_scalar(name = "loss_D",  env= model_name, rate = 1000)
# Training settings
parser = argparse.ArgumentParser(description="PyTorch DenseNet")
parser.add_argument("--batch_size", type=int, default = 5, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=100, help="number of epochs to train for")
parser.add_argument("--decay_epoch", type=int, default=30, help="lr start to decay linearly from decay_epoch")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")

parser.add_argument("--lr", type=float, default=1e-2, help="Learning Rate. Default=1e-4")
parser.add_argument("--cuda", default=True, action="store_false", help="Use cuda?")
parser.add_argument("--resume", default=True, help="Path to checkpoint (default: none)")
parser.add_argument("--reload_epoch", default=53, type=int, help="Manual epoch number (useful on restarts)")

parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--save_freq", default=1, type=int, help="save frequency")
parser.add_argument("--img_size", default=256, type=int, help="image size.")


opt = parser.parse_args()
print(opt)

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

opt.seed = random.randint(1, 10000)
print(("Random Seed: ", opt.seed))
torch.manual_seed(opt.seed)
if cuda:
    import torch.backends.cudnn as cudnn
    torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

print("===> Building model")
netG_A2B = Generator()
netG_B2A = Generator()
netD_A = Discriminator()
netD_B = Discriminator()
l1_criterion = L1_Charbonnier_loss()
ssim_sim = SSIM()

criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()
    l1_criterion.cuda()
    ssim_sim.cuda()

print("===> Training")
netG_A2B.train()
netG_B2A.train()
netD_A.train()
netD_B.train()


# optionally resume from a checkpoint
if opt.resume is True:
    netG_A2B_model_path = os.path.join(model_folder, 'netG_A2B_model_epoch_{}.pth'.format(opt.reload_epoch))
    netG_B2A_model_path = os.path.join(model_folder, 'netG_B2A_model_epoch_{}.pth'.format(opt.reload_epoch))
    netD_A_model_path = os.path.join(model_folder, 'netD_A_model_epoch_{}.pth'.format(opt.reload_epoch))
    netD_B_model_path = os.path.join(model_folder, 'netD_B_model_epoch_{}.pth'.format(opt.reload_epoch))

    if os.path.isfile(netG_A2B_model_path):
        print(("=> loading checkpoint '{}'".format(netG_A2B_model_path)))
        netG_A2B_model_state = torch.load(netG_A2B_model_path)
        netG_A2B.load_state_dict(netG_A2B_model_state)

        netG_B2A_model_state = torch.load(netG_B2A_model_path)
        netG_B2A.load_state_dict(netG_B2A_model_state)
        
        netD_A_model_state = torch.load(netD_A_model_path)
        netD_A.load_state_dict(netD_A_model_state)

        netD_B_model_state = torch.load(netD_B_model_path)
        netD_B.load_state_dict(netD_B_model_state)

        opt.start_epoch =   opt.reload_epoch + 1

    else:
        print(("=> no checkpoint found at '{}'".format(opt.resume)))
    
print("===> Setting Optimizer")

optimizer_G      = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                    lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A    = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B    = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

#optimizer_G      = torch.optim.SGD(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
#                                    lr=opt.lr, momentum = 0.9, nesterov=True)
#optimizer_D_A    = torch.optim.SGD(netD_A.parameters(), lr=opt.lr, momentum = 0.9, nesterov=True)
#optimizer_D_B    = torch.optim.SGD(netD_B.parameters(), lr=opt.lr, momentum = 0.9, nesterov=True)

lr_scheduler_G   = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.nEpochs, opt.start_epoch,  opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.nEpochs, opt.start_epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.nEpochs, opt.start_epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor     = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A    = Tensor(opt.batch_size, 3, opt.img_size, opt.img_size)
input_B    = Tensor(opt.batch_size, 3, opt.img_size, opt.img_size)
input_LOWB = Tensor(opt.batch_size, 3, opt.img_size, opt.img_size)

target_real = Variable(Tensor(opt.batch_size, 1, 1,1).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batch_size, 1, 1, 1).fill_(0.0), requires_grad=False)
 
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()


print("===> Loading datasets")
#train_set = DatasetFromHdf5("/path/to/your/dataset/like/imagenet_50K.h5")
home = os.path.expanduser('~')
proj_root = os.path.expanduser('..')
hd_folder  = os.path.join(proj_root, 'data', 'HD')
reg_folder = os.path.join(proj_root, 'data', 'Regular')
#hd_folder = os.path.join('data', 'HD')
training_data_loader = DataSet(hd_folder, reg_folder=reg_folder, batch_size = opt.batch_size, img_size = opt.img_size)


for epoch in range(opt.start_epoch, opt.nEpochs + 1): 
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    for iteration in range(training_data_loader.epoch_iteration):
        
        batch_data = training_data_loader.get_next()
        ###----------------------------The following is for GAN-------------------------------
        # Set model input
        real_A = Variable(input_A.copy_( torch.from_numpy(batch_data['A'])   ))
        real_B = Variable(input_B.copy_( torch.from_numpy(batch_data['B'] )  ))
        low_B  = Variable(input_LOWB.copy_( torch.from_numpy(batch_data['low_B'])  )) 
        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()
        
        # super-resolution loss
        super_B = netG_A2B(low_B)
        super_resolution_loss = l1_criterion(super_B, real_B) - ssim_sim(super_B, real_B)

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        if 0: # i don't need this loss.
            same_B = netG_A2B(real_B)
            b_identity = l1_criterion(same_B, real_B) - ssim_sim(same_B, real_B)
            loss_identity_B = b_identity*5.0
        loss_identity_B = super_resolution_loss*5.0

        #loss_identity_B = criterion_identity(same_B, real_B)*5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        a_identity = l1_criterion(same_A, real_A) - ssim_sim(same_A, real_A)
        loss_identity_A = a_identity*5.0
        #loss_identity_A = criterion_identity(same_A, real_A)*5.0
        
        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        #import pdb; pdb.set_trace()
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real.expand(pred_fake.size()) )
        
        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real.expand(pred_fake.size()))

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        cycle_loss_A = l1_criterion(recovered_A, real_A) - ssim_sim(recovered_A, real_A)
        loss_cycle_ABA = cycle_loss_A*10.0
        #loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        cycle_loss_B = l1_criterion(recovered_B, real_B) - ssim_sim(recovered_B, real_B)
        loss_cycle_BAB = cycle_loss_B*10.0
        #loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0
        #super_resolution_loss.backward()
        # loss_identity_A.backward()
        # loss_identity_B.backward()
        # loss_GAN_A2B.backward()
        # loss_GAN_B2A.backward()
        # loss_cycle_ABA.backward()
        # loss_cycle_BAB.backward()
        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
    
        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real.expand(pred_real.size()))

        # Fake loss
        fake_A_pop = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A_pop.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake.expand(pred_fake.size()) )

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real.expand(pred_real.size()) )
        
        # Fake loss
        fake_B_pop = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B_pop.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake.expand(pred_fake.size()))

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################
        
        super_loss_plot.plot(super_resolution_loss.data[0])
        loss_G_identity_plot.plot( (loss_identity_A + loss_identity_B).data[0]  )
        loss_G_GAN_plot.plot((loss_GAN_A2B + loss_GAN_B2A).data[0])
        loss_G_cycle_plot.plot( (loss_cycle_ABA + loss_cycle_BAB).data[0] )
        loss_D_plot.plot( (loss_D_A + loss_D_B).data[0] )

        #------------------------------------------END of GAN training------------------------------------
        if iteration%100 == 0:
            print(  "===> Epoch[{}]({}/{}): super_loss : {:.5f}, loss_G_identity : {:.5f}, loss_G_GAN:{:.5f}, loss_G_cycle:{:.5f}, loss_D: {:.5f}".format( 
                    epoch, iteration, training_data_loader.epoch_iteration, 
                    super_resolution_loss.data[0],
                    (loss_identity_A + loss_identity_B).data[0], # loss_G_identity
                    (loss_GAN_A2B + loss_GAN_B2A).data[0],         # loss_G_GAN 
                    (loss_cycle_ABA + loss_cycle_BAB).data[0],     # loss_G_cycle
                    (loss_D_A + loss_D_B).data[0]  )               # loss_D  
                   )
            #print("total gradient", total_gradient(model.parameters()))  
            
            reg_img_np   = batch_data['A'][0:1]
            hd_img_np    = batch_data['B'][0:1]
            low_b_img_np = batch_data['low_B'][0:1]
            recoverd_lowb_img_np = super_B.data.cpu().numpy()[0:1]
            recoverd_reg_img_np  = fake_B.data.cpu().numpy()[0:1]
            
            reg_hd_list   = [reg_img_np, recoverd_reg_img_np]
            lowb_hd_list  = [hd_img_np,low_b_img_np, recoverd_lowb_img_np]

            reg_display_img = save_images(reg_hd_list, save_path=None, save=False, dim_ordering='th')
            plot_img(X=reg_display_img, win='reg_hd_recovered', env=model_name)
            
            lowb_hd_img = save_images(lowb_hd_list, save_path=None, save=False, dim_ordering='th')
            plot_img(X=lowb_hd_img, win='lowb_hd_img', env=model_name)

    # save the checkpoints every epoch        

    if epoch > 0 and epoch % opt.save_freq == 0:
        torch.save(netG_A2B.state_dict(), os.path.join(model_folder, 'netG_A2B_model_epoch_{}.pth'.format(epoch)))
        torch.save(netG_B2A.state_dict(), os.path.join(model_folder, 'netG_B2A_model_epoch_{}.pth'.format(epoch)))
        torch.save(netD_A.state_dict(), os.path.join(model_folder, 'netD_A_model_epoch_{}.pth'.format(epoch)))
        torch.save(netD_B.state_dict(), os.path.join(model_folder, 'netD_B_model_epoch_{}.pth'.format(epoch)))
        
        print('save weights at {}'.format(model_folder))
        