import os
import argparse
import random
import json

import torch
import torchvision.utils as vutils

from td_models import TdAE, AELoss
from dataloader import get_loader


def write_config(opt, step):
    with open(os.path.join(opt.savepath, 'experiments/%s/params/%d.cfg'%(opt.model_name, step)), 'w') as f:
        f.write(json.dumps(vars(opt)))


def save_models(model, basepath, step):
    torch.save(model.state_dict(), os.path.join(basepath, 'gen_%d.pth'%step))


def five_dim_to_img_seq(vector, opt, step):
    size = tuple(vector.size())
    vector = vector.permute(0, 2, 1, 3, 4)
    for b in range(size[0]):
        for c in range(size[1]):
            img_vec = vector[b, c, :, :, :]
            vutils.save_image(img_vec.data, '%s/%s_%s_%s-%s.png'%(opt.savepath, opt.model_name, str(step), str(b), str(c)), normalize=True)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    creterion = AELoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = creterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, opt, step):
    for batch_idx, (data, target) in enumerate(test_loader):
        data, _ = data.to(device), target.to(device)
        output = model(data)

        five_dim_to_img_seq(output, opt, step)

        print('Test Epoch: [{}/{} ({:.0f}%)]'.format(
            batch_idx * len(data), len(test_loader.dataset), 100. * batch_idx / len(test_loader)))
    

def main():

    # args parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='../dataset/solar/', type=str)
    parser.add_argument('--test_dataset', default='../test/solar/', type=str)
    parser.add_argument('--savepath', default='output/solar/', type=str)
    parser.add_argument('--model_name', default='AE', type=str)
    parser.add_argument('--batch', default=8, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--in_seq', default=3, type=int)
    parser.add_argument('--out_seq', default=3, type=int)
    parser.add_argument('--channel', default=1, type=int)
    parser.add_argument('--hidden', default=1024, type=int)
    parser.add_argument('--image_scale', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--manual_seed', default=826, type=int)
    parser.add_argument('--gpuid', default=0, type=int)
    parser.add_argument('--cuda', action='store_true')
    opt = parser.parse_args()

    # set random seed
    print("Random Seed: ", opt.manual_seed)
    random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    # cuda setting
    if opt.cuda:
        opt.cuda = True
        torch.cuda.set_device(opt.gpuid)
        torch.cuda.manual_seed_all(opt.manual_seed)
        model = TdAE(opt.in_seq, opt.out_seq, opt.channel, opt.hidden)
        model = torch.nn.DataParallel(model, device_ids=[opt.gpuid])

    # save path
    parampath = os.path.join(opt.savepath, 'experiments/%s/params/' % opt.model_name)
    for path in [opt.savepath, parampath]:
        if not os.path.exists(path):
            os.makedirs(path)

    kwargs = {'num_workers': 1, 'pin_memory': True} if opt.cuda else {}
    train_loader = get_loader(opt.dataset, batch_size=opt.batch, seq_size=opt.in_seq, scale_size=opt.image_scale)
    test_loader = get_loader(opt.test_dataset, batch_size=opt.batch, seq_size=opt.in_seq, scale_size=opt.image_scale)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    
    for epoch in range(1, opt.epochs + 1):
        # train
        train(model, 'cuda', train_loader, optimizer, epoch)

        # save
        if epoch % 10 == 0:
            save_models(model, opt.savepath, epoch)
            write_config(opt, epoch)
            print('Model Saved')
            test(model, 'cuda', test_loader, opt, epoch)


if __name__ == '__main__':
    main()