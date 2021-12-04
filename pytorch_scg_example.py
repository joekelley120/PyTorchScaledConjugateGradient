import torch
from scg import SCG
import matplotlib.pyplot as plt
import math


def main():

    x = torch.unsqueeze(torch.linspace(-2, 2, 200), dim=1)
    y = torch.sin(x * 2 * math.pi) + 0.2*torch.rand(x.size())
    x, y = torch.autograd.Variable(x), torch.autograd.Variable(y)

    # Define a network for regression
    class Regression(torch.nn.Module):
        def __init__(self, n_feature, n_hidden, n_output):
            super(Regression, self).__init__()
            self.hidden = torch.nn.Linear(n_feature, n_hidden)
            self.predict = torch.nn.Linear(n_hidden, n_output)

        def forward(self, input):
            input = torch.tanh(self.hidden(input))
            input = self.predict(input)
            return input

    net = Regression(n_feature=1, n_hidden=40, n_output=1)
    optimizer = SCG(net.parameters())
    loss_func = torch.nn.MSELoss()

    fig, ax = plt.subplots(figsize=(8, 5))

    # train the network
    for i in range(1000):

        def closure():
            optimizer.zero_grad()
            pred = net(x)
            loss = loss_func(pred, y)
            loss.backward()
            return loss

        loss = optimizer.step(closure)

        if (i + 1) % 25 == 0 or i + 1 == 1:
            print("Iteration: " + str(i + 1) + ", Loss: " + f'{loss:.9f}')

            plt.cla()
            ax.set_title('Regression Analysis - Iteration: ' + str(i + 1) + ' - Loss: ' + f'{loss:.9f}')
            ax.set_xlabel('Inputs')
            ax.set_ylabel('Outputs')
            ax.scatter(x.data.numpy(), y.data.numpy(), color="orange")
            ax.plot(x.data.numpy(), net(x).data.numpy(), 'g-', lw=3)
            plt.pause(0.0005)

    plt.show()


if __name__ == "__main__":
    main()