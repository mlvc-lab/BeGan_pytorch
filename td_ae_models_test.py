import torch
from torch.autograd import Variable
import unittest

from td_ae_models import TdEncoder, TdDecoder, TdAE


class Td_Model_Test(unittest.TestCase):

    def setUp(self):
        pass

    def test_encoder(self):
        batch_size = 1
        input_size = (64, 64)
        input_channel = 1
        input_seq = 3
        hidden_size = 10

        en = TdEncoder(input_seq, input_channel, hidden_size)

        input = Variable(torch.rand(batch_size, input_seq, input_channel, input_size[0], input_size[1]))
        out = en(input)

        self.assertEqual(tuple(out.size()), (batch_size, 10))

    def test_decoder(self):
        batch_size = 1
        output_channel = 1
        output_seq = 3
        hidden_size = 10

        de = TdDecoder(output_seq, output_channel, hidden_size)

        input = Variable(torch.rand(batch_size, 10))
        out = de(input)

        self.assertEqual(tuple(out.size()), (batch_size, output_seq, output_channel, 64, 64))

    def test_TdAE(self):
        batch_size = 1
        input_size = (64, 64)
        input_channel = 1
        input_seq = 10
        output_seq = 3
        hidden_size = 10

        ae = TdAE(input_seq, output_seq, input_channel, hidden_size)

        input = Variable(torch.rand(batch_size, input_seq, input_channel, input_size[0], input_size[1]))
        out = ae(input)

        self.assertEqual(tuple(out.size()), (batch_size, output_seq, input_channel,  input_size[0], input_size[1]))


if __name__ == '__main__':
    unittest.main()
