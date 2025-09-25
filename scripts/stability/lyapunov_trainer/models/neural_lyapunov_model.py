import torch

class NeuralLyapunovModel(torch.nn.Module):
    
    def __init__(self, n_input, hidden_layers=None):
        super(NeuralLyapunovModel, self).__init__()
        if hidden_layers is None:
            # simple 2 layer MLP
            hidden_layers = [64, 64]
        self.input_layer = torch.nn.Linear(n_input, hidden_layers[0])
        self.hidden = torch.nn.ModuleList()
        for k in range(len(hidden_layers)-1):
            layer = torch.nn.Linear(hidden_layers[k], hidden_layers[k+1])
            self.hidden.append(layer)
        self.output_layer = torch.nn.Linear(hidden_layers[-1], 1)
        self.activation = torch.nn.Tanh()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden:
            x = self.activation(layer(x))
        # V is candidate lyapunov function
        V = self.activation(self.output_layer(x))
        return V

if __name__ == '__main__':
    # test out function

    # number of samples
    N = 1
    # inputs 
    D_in = 4
    hidden_layers = [6]

    x = torch.Tensor(N, D_in).uniform_(-6, 6)       

    model = NeuralLyapunovModel(D_in, hidden_layers)
    V = model(x)

    print(V)
