from td_models import TdEncoder, TdDecoder
import torch
from torch.autograd import Variable


def test_encoder():
    print("=====Test Encoder=====")
    batch_size = 1
    input_size = (64, 64)
    input_channel = 1
    input_seq = 3
    hidden_size = 10

    en = TdEncoder(input_seq, input_channel, hidden_size)
    print(en)

    input = Variable(torch.rand(batch_size, input_seq, input_channel, input_size[0], input_size[1]))
    out = en(input)

    print('in', input.size())
    print('out', out.size())


def test_decoder():
    print("=====Test Decoder=====")
    batch_size = 1
    input_size = (64, 64)
    output_channel = 1
    output_seq = 3
    hidden_size = 10

    de = TdDecoder(output_seq, output_channel, hidden_size)
    print(de)

    input = Variable(torch.rand(batch_size, 10))
    out = de(input)

    print('in', input.size())
    print('out', out.size())


if __name__ == '__main__':
    test_encoder()
    test_decoder()
