import torch
from torch.nn import Embedding, LSTM, Linear, Dropout, ReLU, BatchNorm1d

class ProteinSequenceClassifierModel(torch.nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout_val=None):
    super(ProteinSequenceClassifierModel, self).__init__()

    self.hidden_dim = hidden_dim
    self.embedding_dim = embedding_dim
    self.vocab_size = vocab_size
    self.output_dim = output_dim
    self.n_layers = n_layers
    self.dropout_val = dropout_val

    
    self.embedding_layer = Embedding(vocab_size, embedding_dim)
    self.LSTM_layer = LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True)
    num_directions = 2 if self.LSTM_layer.bidirectional else 1

    self.fc_1 = Linear(hidden_dim*num_directions, hidden_dim)

    self.fc_2 = Linear(hidden_dim, output_dim)


    self.embedding_layer.cuda()
    self.LSTM_layer.cuda()
    self.fc_1.cuda()
    self.fc_2.cuda()


    #self.embedding_layer.requires_grad_ = True
    #self.LSTM_layer.requires_grad_ = True
    #self.fc_1.requires_grad_ = True
    #self.fc_2.requires_grad_ = True

  def forward(self, input):

    # feed input into embedding layer
    embedded = self.embedding_layer(input)
    embedded.cuda()

    # get output and hidden state from lstm layer
    num_directions = 2 if self.LSTM_layer.bidirectional else 1

    hidden_state = (torch.randn(num_directions*self.n_layers,input.size(0),self.hidden_dim).cuda(), torch.randn(num_directions*self.n_layers,input.size(0),self.hidden_dim).cuda())


    out, v = self.LSTM_layer(embedded, hidden_state)

    out.cuda()

    out = self.fc_1(out)

    
    # try sigmoid in between?
    out = torch.sigmoid(out)

    # second dense layer
    out = self.fc_2(out)


    out = torch.sigmoid(out)

    return out