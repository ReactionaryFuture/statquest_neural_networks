
# To follow along with this code navigate to repo directory in console and
# run the following command "python3 -i neural_networks_pt_1.py"
# You will then be able to evalue functions as we go along in Python IDLE.

import math as m
import matplotlib.pyplot as plt


#############
# Section 2 #
#############

# Lets imagine we designed a drug and we gave the drug to three different groups
# of people with three different dosages.

#   Yes(1)  |           ooo
#     ^     |
# Efficancy |
#     v     |
#   No(0)   |ooo___________________ooo          
#            Low(0) < Dosage > High(1) 

dosages    = [0.01, 0.02, 0.03, 0.49, 0.50, 0.51, 0.97, 0.98, 0.99]
efficancys = [0.01, 0.02, 0.03, 0.98, 0.99, 0.98, 0.03, 0.02, 0.01]
questions       = dosages
correct_answers = efficancys

# Now that we have this data we would like to use it to predict whether or not
# future dosages will be effective.
def predict_future_dosage_effectiveness(future_dosage):
    # However we cant just fit a straight line to the data to make predictions
    # because no matter how we rotate the straight line it can only accurately
    # predict 2 of the 3 dosages.
    # The good news is that a neural network can fit a squiggle to the data.
    question = future_dosage
    return neural_network_predictions(question)


#############
# Section 3 #
#############

# A neural network consists of nodes and connections between the nodes.
class Node():
    
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias    = bias

        
# These values come from the video @T=3:37
# Note: the numbers along each connection represent paramater values that were
# estimated when this Neural Network was fit to the data
neural_network = [[Node([-34.4      ],  2.14), Node([-2.52], 1.29)],
                  [Node([-1.30, 2.28], -0.58)                     ]]

# There are many common bent or curved lines we can choose for a neural network.
# We are going to use softplus activation function. 
def softplus(x):
    return m.log(1 + m.e ** x)


# Evaluate this function "graph_softplus()" in IDLE to see the curve.
def graph_softplus():
    # Here we are just making a list of Xs 1000 elements long starting at -5, 
    # finishing at 5 and stepping 0.01. We are going to use this a few times.
    Xs = [(i-500)/100 for i in range(1000)]
# Here we make a list of Ys Xs elements long where each Y is f(x) and f
    # is softplus.
    Ys = [softplus(x) for x in Xs]
    # Plot Xs and Ys
    plt.scatter(Xs, Ys, color='black'); plt.show()


#############
# Section 4 #
#############

# @~T=8:17
# Lets make this function that takes a neuron and an input and return how 
# activated it is. You know, this => softplus(sum([weights * inputs]) + bias)
def activate_neuron(inputs, neuron, act_func):
    # multiply each weight times each activation from previous layer
    weighted_inputs = [i * w for i, w in zip(inputs, neuron.weights)]
    # sum and add the bias
    output = sum(weighted_inputs) + neuron.bias
    # apply the softplus activation function when act_func flag set true
    if act_func: output = softplus(output)
    return output

# From this point forward for atleast a little while curves will be reffered to
# by their color in the video and commentary from myself will be kept to a
# minimum. Follow closely along with the video and evaluate these fuctions when
# appropreate. 

# We'll use these values a lot. Lets just assign these once.
nn = neural_network
Xs = [i/1000 for i in range(1000)]

# @T=10:15
def graph_blue_1(*args):
    Ys = [activate_neuron([x], nn[0][0], act_func=True) for x in Xs]
    plt.scatter(Xs, Ys, color='deepskyblue')
    for ar in args: 
        if ar == 1: return
    plt.show()

# @T=12:08
def graph_blue_2(*args):
    graph_blue_1(1)
    Ys = [activate_neuron([x], nn[0][0], act_func=True ) for x in Xs]
    Ys = [y * -1.3 for y in Ys]         
    plt.scatter(Xs, Ys, color='deepskyblue'); 
    for ar in args: 
        if ar == 1: return
    plt.show()

# @T=13:32
def graph_orange_1(*args):
    graph_blue_2(1)
    Ys = [activate_neuron([x], nn[0][1], act_func=True) for x in Xs]
    plt.scatter(Xs, Ys, color='orange')
    for ar in args: 
        if ar == 1: return
    plt.show()

# @T=14:51
def graph_orange_2(*args):
    Ys = [activate_neuron([x], nn[0][1], act_func=True) for x in Xs]
    Ys = [y * 2.28 for y in Ys]         
    plt.scatter(Xs, Ys, color='orange')
    for ar in args: 
        if ar == 1: return
    plt.show()

# @T=14.59
# Lets define a forward propagation for the whole network called
# neural_network_predictions(). The function is deisgned to take question as a
# list to make it genralizable to multiple inputs in the future. Inorder to
# take a single input it needs to simply be passed a singleton list.
def neural_network_prediction(question):
    l1a = [activate_neuron(question, n, act_func=True ) for n in nn[0]]
    l2a = [activate_neuron(      l1a, n, act_func=False) for n in nn[1]]
    return l2a

# And finally lets graph the whole neural network, combine the previous lines, 
# and see that green squiggle.
def graph_neural_network_precitions():
    graph_blue_2(1)
    graph_orange_2(1)
    Ys = [neural_network_prediction(x) for x in Xs]
    plt.scatter(Xs, Ys, color='green'); plt.show()


#############
# Section 5 #
#############

# @T=15:24
# We can test with 0.5 and see that indeed we get a matching result from the 
# video. The result comes as a list since we will want the ability to receive
# results as lists in the future.
def test_with_zero_point_5():
    print(neural_network_prediction([0.5]))
 
