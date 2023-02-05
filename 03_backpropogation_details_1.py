
# This document is intended as a study companion for "Neural Networks 
# Pt. 2: Backpropagation Main Ideas by StatQuest with Josh Starmer
# URL: https://www.youtube.com/watch?v=IN2XmBhILt4

# To follow along with this code navigate to repo directory in console and
# run the following command "python3 -i 03_backpropogation_details_1.py"
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
    
    def activate(self, inputs, act_func):
        weighted_inputs = [i * w for i, w in zip(inputs, self.weights)]
        output = sum(weighted_inputs) + self.bias
        if act_func: output = softplus(output)
        return output

def softplus(x):
    return m.log(1 + m.e ** x)

nn = [[Node([3.34        ], -1.43), Node([-3.53], 0.57)],
      [Node([-1.22, -2.30],   0.0)                     ]]

def nnp(question):
    l1a = [n.activate(question, act_func=True ) for n in nn[0]]
    l2a = [n.activate(     l1a, act_func=False) for n in nn[1]]
    return = l2a
    
Xs = [i/1000 for i in range(1000)]

def step(observations, questions, alpha):
    predictions = [nnp([q])[0] for q in questions]
    step = nn[1][0].bias - (dssr(observations, predictions) * alpha)
    return step

def gradient_descent(observations, questions, alpha, steps):
    for _ in range(steps):
        nn[1][0].bias = step(observations, questions, alpha)
