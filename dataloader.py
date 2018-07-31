import os
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data


def default_loader(path):
    return Image.open(path).convert('L')


def ToTensor(pic):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backard compability
        return img.float().div(255)
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


# TODO if size of dataset is not divided by seq_size reminder is discarded
class Solar(data.Dataset):
    def __init__(self, dataPath='dataset/solar/', loadSize=64, seqSize=3):
        super(Solar, self).__init__()

        self.input_header = 'AIA0304/'
        self.target_header = 'HMI0100/'
        self.dataPath = dataPath
        self.loadSize = loadSize
        self.seqSize = seqSize
        self.input_images = os.listdir(dataPath+self.input_header)
        self.target_images = os.listdir(dataPath+self.target_header)

    def __getitem__(self, index):
        inputs, targets = [], []

        for idx in range(index, index+self.seqSize):
            input_path = os.path.join(self.dataPath, self.input_header, self.input_images[idx])
            target_path = os.path.join(self.dataPath, self.target_header, self.target_images[idx])

            input_image = default_loader(input_path)
            target_image = default_loader(target_path)
            w, h = input_image.size

            # image resizing
            if h != self.loadSize:
                input_image = input_image.resize((self.loadSize, self.loadSize), Image.BICUBIC)
                target_image = target_image.resize((self.loadSize, self.loadSize), Image.BICUBIC)

            input_image = ToTensor(input_image)  # C x W x H
            target_image = ToTensor(target_image)

            input_image = input_image.mul_(2).add_(-1)
            target_image = target_image.mul_(2).add_(-1)

            inputs.append(input_image)
            targets.append(target_image)

        return np.stack(inputs), np.stack(targets)

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.input_images) - self.seqSize + 1


def get_loader(root, batch_size, scale_size=64, seq_size=3, num_workers=12):
    dataset = Solar(root, scale_size, seq_size)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers)
    return data_loader