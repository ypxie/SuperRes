import argparse, os
import pdb
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from srdense.thinnet import thinnet as srnet
from srdense.thinnet import  L1_Charbonnier_loss
from srdense.ssim import SSIM


from srdense.dataset import DatasetFromfolder
from srdense.proj_utils.plot_utils import *


model_name = 'thinnet_super_resolution'
model_folder = os.path.join('model_adam', model_name) 
if not os.path.exists(model_folder):
    os.mkdir(model_folder)

l1_plot = plot_scalar(name = "l1_loss_sr",  env= model_name, rate = 1000)
ssim_plot = plot_scalar(name = "ssim_loss_sr",  env= model_name, rate = 1000)

# Training settings
parser = argparse.ArgumentParser(description="PyTorch DenseNet")
parser.add_argument("--batch_size", type=int, default=20, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=1000, help="number of epochs to train for")
parser.add_argument("--lr",  type=float, default=1e-2, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=30, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", default=True, action="store_false", help="Use cuda?")
parser.add_argument("--resume", default=True, help="Path to checkpoint (default: none)")
parser.add_argument("--reload_epoch", default=100, type=int, help="Manual epoch number (useful on restarts)")

parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--save_freq", default=1, type=int, help="save frequency")

def main():

    global opt, model 
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
    model = srnet()
    criterion = L1_Charbonnier_loss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        ssim_sim  = SSIM().cuda()
        
    # optionally resume from a checkpoint
    if opt.resume is True:
        model_path = os.path.join(model_folder, 'model_epoch_{}.pth'.format(opt.reload_epoch))

        if os.path.isfile(model_path):
            print(("=> loading checkpoint '{}'".format(model_path)))
            model_state = torch.load(model_path)
            model.load_state_dict(model_state)
            opt.start_epoch =   opt.reload_epoch + 1

        else:
            print(("=> no checkpoint found at '{}'".format(opt.resume)))
        
    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print(("=> loading model '{}'".format(opt.pretrained)))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print(("=> no model found at '{}'".format(opt.pretrained))) 
            
    print("===> Setting Optimizer")
    #optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    optimizer    = optim.SGD(model.parameters(), lr=opt.lr, momentum = 0.9, nesterov=True)

    print("===> Training")
    model.train()  

    print("===> Loading datasets")
    #train_set = DatasetFromHdf5("/path/to/your/dataset/like/imagenet_50K.h5")
    home = os.path.expanduser('~')
    hd_folder = os.path.join(home, 'DataSet', 'SR_DATA', 'HD')
    #hd_folder = os.path.join('data', 'HD')

    training_data_loader = DatasetFromfolder(hd_folder, batch_size = opt.batch_size, img_size = 256)


    for epoch in range(opt.start_epoch, opt.nEpochs + 1): 
        #train(training_data_loader, optimizer, model, criterion, epoch)

        # lr = adjust_learning_rate(optimizer, epoch-1)
        lr = opt.lr * (0.1 ** (( epoch-1) // opt.step))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr  
        print("epoch =", epoch,"lr =",optimizer.param_groups[0]["lr"])

        for iteration in range(training_data_loader.epoch_iteration):
            
            batch_data = training_data_loader.get_next()

            inputs = Variable(torch.from_numpy(batch_data[0]), requires_grad=False)
            label  = Variable(torch.from_numpy(batch_data[1]), requires_grad=False)

            if opt.cuda:
                inputs = inputs.cuda()
                label = label.cuda()
            #print(inputs.size())
            out = model(inputs)
            
            l1_loss   = criterion(out, label) 
            ssim_loss = - ssim_sim(out, label)
            loss = l1_loss + ssim_loss

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            
            l1_plot.plot(l1_loss.cpu().data.numpy()[0])
            ssim_plot.plot(ssim_loss.cpu().data.numpy()[0])

            if iteration%100 == 0:
                print(("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, training_data_loader.epoch_iteration, loss.data[0])))
                #print("total gradient", total_gradient(model.parameters()))  
                
                reg_img_np = batch_data[0][0:1]
                hd_img_np = batch_data[1][0:1]
                recoverd_img_np = out.data.cpu().numpy()[0:1]
                img_disply = [reg_img_np, hd_img_np, recoverd_img_np]

                returned_img = save_images(img_disply, save_path=None, save=False, dim_ordering='th')
                plot_img(X=returned_img, win='reg_hd_recovered', env=model_name)
                
        # save the checkpoints every epoch        

        if epoch > 0 and epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(model_folder, 'model_epoch_{}.pth'.format(epoch)))
            print('save weights at {}'.format(model_folder))
            

#def train(training_data_loader, optimizer, model, criterion, epoch):


if __name__ == "__main__":
    main()
