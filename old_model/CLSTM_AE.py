import torch
import torch.nn as nn
from torch.autograd import Variable
from old_model.ConvLSTM_model import CLSTM, weights_init


class Encoder(nn.Module):
    def __init__(self, batch_size, input_size, input_dim, kernel_size, hidden_dim, num_layers, isCuda=False):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.input_dim = input_dim
        self.hidden_size = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        self.isCuda = isCuda
        self.lstm = CLSTM(input_size, input_dim, kernel_size, hidden_dim, num_layers)
        self.lstm.apply(weights_init)
        if isCuda:
            self.lstm.cuda()
        self.activation = nn.ReLU()

    def forward(self, input, hidden_state):
        tt = torch.cuda if self.isCuda else torch
        encoded_input, hidden = self.lstm(input, hidden_state)
        encoded_input = self.activation(encoded_input)
        return encoded_input


class Decoder(nn.Module):
    def __init__(self, batch_size, input_size, input_dim, hidden_dim, kernel_size, num_layers, isCuda):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.input_dim = input_dim
        self.hidden_size = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        self.isCuda = isCuda
        self.lstm = CLSTM(input_size, input_dim, hidden_dim, kernel_size, num_layers)
        if isCuda:
            self.lstm.cuda()
        self.lstm.init_hidden(self.batch_size)
        self.activation = nn.Sigmoid()

    def forward(self, encoded_input, hidden_state):
        tt = torch.cuda if self.isCuda else torch
        decoded_output, hidden = self.lstm(encoded_input, hidden_state)
        decoded_output = self.activation(decoded_output)
        return decoded_output


class LSTMAE(nn.Module):
    def __init__(self, batch_size, input_size, input_dim, hidden_dim, kernel_size, num_layers, isCuda):
        super(LSTMAE, self).__init__()
        self.encoder = Encoder(batch_size, input_size, input_dim, hidden_dim, kernel_size, num_layers, isCuda)
        self.decoder = Decoder(batch_size, input_size, hidden_dim, input_dim, kernel_size, num_layers, isCuda)

    def forward(self, input, hidden_state):
        encoded_input, hidden = self.encoder(input, hidden_state)
        decoded_output, hidden = self.decoder(encoded_input, hidden)
        return decoded_output, hidden
    

if __name__ == '__main__':
    batch_size = 10
    input_size = (25, 25)
    input_dim = 3
    hidden_dim = 16
    kernel_size = 5
    num_layers = 2
    isCuda = True
    
    seq_len = 4
    
    model = LSTMAE(batch_size, input_size, input_dim, hidden_dim, kernel_size, num_layers, isCuda)
    
    inputs = Variable(torch.rand(batch_size, seq_len, input_dim, input_size[0], input_size[1]))
    hidden_state = model.encoder.lstm.init_hidden(batch_size)
    
    out = model(inputs, hidden_state)
    
    print(out.size())
    print(out)
        