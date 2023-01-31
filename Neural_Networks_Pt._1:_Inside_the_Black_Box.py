import math as m


class Neuron():

    def __init__(self, weight, bias):
        self.w = weight
        self.b = bias

    def softplus(self, input):
        return m.log(1.0 + m.e ** input)

    def activate(self, input, act_func):
        weighted_inputs = [i * w for i, w in zip(input, self.w)]
        output = sum(weighted_inputs + [self.b])
        if act_func: output = self.softplus(output)
        return output


def fprop(input):
    l1a = [n.activate(input, True) for n in nn[0]]
    l2a = [n.activate(l1a, False) for n in nn[1]]
    return l2a

def plot():
    Xs = []
    Ys = []

    for i in range(100):
        j = i / 100
        Xs.append(j)
        Ys.append(fprop([j]))

    import matplotlib.pyplot as plt
    plt.scatter(Xs, Ys, color='green')
    plt.show()

nn = [[Neuron([ 3.34      ], -1.43), Neuron([-3.53], 0.57)],
      [Neuron([-1.22, -2.3], -0.58)                       ]]

plot()
