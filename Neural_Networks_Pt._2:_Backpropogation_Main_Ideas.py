import math as m
import matplotlib.pyplot as plt


class Neuron():

    def __init__(self, weight, bias):
        self.weight = weight
        self.bias   = bias

    def softplus(self, input):
        return m.log(1.0 + m.e ** input)

    def activate(self, input, act_func):
        # multiply each weight times each activation from previous layer
        weighted_inputs = [i * w for i, w in zip(input, self.weight)]
        # sum and add the bias
        output = sum(weighted_inputs) + self.bias
        # apply the softplus activation function when act_func flag set true
        if act_func: output = self.softplus(output)
        return output


def forward_prop(input):
    # lists of activation values for each layer of neural net
    l1a = [n.activate(input, act_func=True ) for n in nn[0]]
    l2a = [n.activate(  l1a, act_func=False) for n in nn[1]]
    return l2a

def sum_of_square_residuals(input):
    # differences between observed data and predicted squared and summed
    return sum([(o - forward_prop(p)[0])**2 for o, p in input])

# innitalize neural net with values from video
# nn[0] is layer 1 and nn[1] is layer 2
nn = [[Neuron([ 3.34      ], -1.43), Neuron([-3.53], 0.57)],
      [Neuron([-1.22, -2.3],     0)                       ]]
# values from video for observed data and inputs for neural network predictions
observed = [(0, [0.0]), (1, [0.5]), (0, [1.0])]

def plot():
    # Plot x on x axis for each 100th between 0 and 1 but this timewe hold the 
    # input of forward_prop constant and adjust bias of layer 2 at each step.
    Xs = []; Ys = []
    for i in range(100):
        nn[1][0].bias = i/20
        Xs.append(i/100)
        Ys.append(sum_of_square_residuals(observed))
    plt.scatter(Xs, Ys, color='deeppink')
    plt.show()

# double bam
plot()
