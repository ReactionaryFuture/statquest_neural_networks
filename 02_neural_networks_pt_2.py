
# This document is intended as a study companion to go along with "Neural
# Networks Pt. 2: Backpropagation Main Ideas by StatQuest with Josh Starmer
# URL: https://www.youtube.com/watch?v=IN2XmBhILt4

# To follow along with this code navigate to repo directory in console and
# run the following command "python3 -i 02_backpropagation_main_ideas"
# You will then be able to evalue functions as we go along in Python IDLE.

import math as m
import matplotlib.pyplot as plt

# Lets bring over some of the code from 01_inside_the_black_box.py with a few
# changes. This should look pretty familiar and will be presented mostly without
# commentary. Review part one if any of this looks confusing.

class Node():
    
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias    = bias
    
    # The activate_neuron function is going to become a method inside of our 
    # neuron class and have its name changed to simply activate since it's 
    # part of neuron.
    def activate(self, inputs, act_func):
        weighted_inputs = [i * w for i, w in zip(inputs, self.weights)]
        output = sum(weighted_inputs) + self.bias
        if act_func: output = softplus(output)
        return output

def softplus(x):
    return m.log(1 + m.e ** x)

# New values in part 2.
nn = [[Node([3.34        ], -1.43), Node([-3.53], 0.57)],
      [Node([-1.22, -2.30],   0.0)                     ]]

# Neural network prediction.
def nnp(question):
    l1a = [n.activate(question, act_func=True ) for n in nn[0]]
    l2a = [n.activate(     l1a, act_func=False) for n in nn[1]]
    prediction = l2a
    return prediction

Xs = [i/1000 for i in range(1000)]


#############
# Section 1 #
#############

# @T=0:57
# Lets see the curve for our new neural network with fitted weights and biases.
def graph_new_nn():
    nn[1][0].bias = 2.61
    Ys = [nnp([x]) for x in Xs]
    plt.scatter(Xs, Ys, color='green'); plt.show()
    nn[1][0].bias = 0


#############
# Section 2 #
#############

# @T=5:21
# If we graph the curve with the third bias (b sub 3) at 0 we get this result.
def graph_with_bias_zero():
    Ys = [nnp([x]) for x in Xs]
    plt.scatter(Xs, Ys, color='green'); plt.show()


#############
# Section 3 #
#############

# Lets define some dosages, efficiencies and predictions.
dosages      = [0.0, 0.5, 1.0]
efficancys   = [0.0, 1.0, 0.0]
# I awant to make dosages and effiencies into something more general that we can
# use throught to conceptualize the behavior of the neural network. I want to
# think of the network as taking questions, returning predictions and the 
# correct answers (perfect prediction) as observations.
questions    = dosages 
observations = efficancys
predictions  = [nnp([q])[0] for q in questions]

# @T=5:57
# Lets define sum of the squared residuals. (from here on SSR)
# Here is mathmatical notation
#
#       n
# SSR = Σ(Observedᵢ - Predictedᵢ)^2
#       i
#
# and here it is in code.
def ssr(observations, predictions):
    ssr = sum([(o-p)**2 for o, p in zip(observations, predictions)])
    return ssr


#############
# Section 4 #
#############

# @T=7:10
# And lets graph the pink curve which is the SSR with respect to b sub 3.
def graph_pink():
    Xs = [i/100 for i in range(400)]; Ys = []
    for x in Xs:
        nn[1][0].bias = x
        Ys.append(ssr(observations, nnp(observations)))
    plt.scatter(Xs, Ys, color='deeppink'); plt.show()
    nn[1][0].bias = 0.0


#############
# Section 5 #
#############

# @T=8:27
# Now lets define the derivative of the SSR with respect to b sub 3. (from here
# on dSSR)
# Here is in math language.
#
#        n
# dSSR = Σ(-2)(Observedᵢ - Predictedᵢ)
#        i
#
# And here it is in code:
def dssr(observations, predictions):
    dssr = sum([-2*(o-p) for o, p in zip(observations, predictions)])
    return dssr


#############
# Section 6 #
#############

# @T=13:29
# We can use the dSSR, the bias, and a learning rate to take a step towards 
# optizing the bias with respect to the SSR. We do this by multiplying the dSSR 
# times the learning rate and subtracting it from the current bias.
def step(observations, questions, alpha):
    predictions = [nnp([q])[0] for q in questions]
    step = nn[1][0].bias - (dssr(observations, predictions) * alpha)
    return step

# And finally we can make gradient descent! Note: we must recalculate 
# predictions with each iteration since we need a new dSSR with each step to 
# take those gradual controlled steps towards optimization that we are looking 
# for in b sub 3, and a new dSSR requires a new SSR which requires new residuals
# which requires new predictions.
def gradient_descent(observations, questions, alpha, steps):
    for _ in range(steps):
        nn[1][0].bias = step(observations, questions, alpha)

# D... I mean... Baaam!
def test_gradient_desent():
    nn[1][0].bias = 0
    gradient_descent(observations, questions, 0.1, 8)
    print(nn[1][0].bias)
    
