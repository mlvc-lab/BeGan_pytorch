import os
import argparse
import random

import torch
from td_models import TdAE, AELoss
from dataloader import get_loader


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    creterion = AELoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = creterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def main():

    # args parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='../dataset/solar/', type=str)
    parser.add_argument('--batch', default=16, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--in_seq', default=3, type=int)
    parser.add_argument('--out_seq', default=3, type=int)
    parser.add_argument('--channel', default=1, type=int)
    parser.add_argument('--hidden', default=32, type=int)
    parser.add_argument('--image_scale', default=64, type=int)
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

    kwargs = {'num_workers': 1, 'pin_memory': True} if opt.cuda else {}
    train_loader = get_loader(opt.dataset, batch_size=opt.batch, seq_size=opt.in_seq, scale_size=opt.image_scale)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    for epoch in range(1, opt.epochs + 1):
        train(opt, model, 'cuda', train_loader, optimizer, epoch)


if __name__ == '__main__':
    main()