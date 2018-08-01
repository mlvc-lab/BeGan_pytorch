import os
import unittest
from PIL import Image
import matplotlib.pyplot as plt

from dataloader import get_loader


class DataLoaderTest(unittest.TestCase):
    def setUp(self):
        os.mkdir("test")
        os.mkdir("test/AIA0304")
        os.mkdir("test/HMI0100")
        for i in range(50):
            image = Image.new('L', size=(64, 64), color=255)
            image.save("test/AIA0304/test_img{}.jpg".format(i), "JPEG")
            image = Image.new('L', size=(64, 64), color=0)
            image.save("test/HMI0100/test_img{}.jpg".format(i), "JPEG")

    def test_dataloader(self):
        batch_size = 3
        scale_size = 64
        seq_size = 3

        dl = get_loader("test/", batch_size=batch_size, scale_size=scale_size, seq_size=seq_size, num_workers=1)

        for idx, (inputs, targets) in enumerate(dl):
            # print(idx, (inputs.shape, targets.shape))
            # output shape check
            self.assertEqual(tuple(inputs.shape), (batch_size, 1, seq_size, scale_size, scale_size)) 
            self.assertEqual(tuple(targets.shape), (batch_size,  1, seq_size, scale_size, scale_size))
            # implot = plt.imshow(inputs[0, 0, 0, :, :].numpy(), cmap='gray')
            # plt.show()
            # implot = plt.imshow(targets[0, 0, 0, :, :].numpy(), cmap='gray')
            # plt.show()
            # print('input', inputs[0, 0, 0, :, :])
            # print('target', targets[0, 0, 0, :, :])

        # image size and target size check
        self.assertEqual(len(dl.dataset.input_images), len(dl.dataset.target_images))
        # iteration size check
        self.assertEqual(len(dl.dataset.input_images) // dl.dataset.seqSize, idx+1)
        # 
        # x, y = dl.dataset[0]
        # print()
        # plt.imshow(x[0][0], cmap='gray')
        # plt.show()
        # plt.imshow(y[0][0], cmap='gray')
        # plt.show()
        # 
        # print(x[0][0])
        # print(y[0][0])

    def tearDown(self):
        for i in range(50):
            os.remove("test/AIA0304/test_img{}.jpg".format(i).format(i))
            os.remove("test/HMI0100/test_img{}.jpg".format(i).format(i))
        os.rmdir("test/AIA0304")
        os.rmdir("test/HMI0100")
        os.rmdir("test")


if __name__ == '__main__':
    unittest.main()