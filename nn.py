from math import e, log
from random import uniform


def sigmoid(z):
    return 1.0 / (1.0 + e ** -z)


def dsigmoid(a):
    return a * (1 - a)


class Node(object):

    def __init__(self):
        self.weights = []
        self.bias_weight = 0
        self.activation = 0
        self.sum = 0
        self.error = 0

    def activate(self, inputs):
        self.inputs = inputs
        self.sum = 0
        for i in range(len(inputs)):
            self.sum = self.sum + inputs[i] * self.weights[i]
        self.sum = self.sum + self.bias_weight
        self.activation = sigmoid(self.sum)


class Layer(object):

    def __init__(self):
        self.nodes = []

    def activate(self, inputs):
        self.activations = []
        for node in self.nodes:
            node.activate(inputs)
            self.activations.append(node.activation)
        return self.activations


class NeuralNetwork(object):

    def __init__(self):
        self.layers = []
        self.input_size = 0

    def set_input_size(self, n):
        self.input_size = n

    def add_layer(self, layer, gen_weights=True):
        if gen_weights:
            if len(self.layers) == 0:
                input_size = self.input_size
            else:
                input_size = len(self.layers[-1].nodes)

            for node in layer.nodes:
                weight = uniform(-0.2, 0.2)
                node.weights = [weight for _ in range(input_size)]
                node.bias_weight = weight
        self.layers.append(layer)

    def activate(self, inputs):
        activation = inputs

        for layer in self.layers:
            activation = layer.activate(activation)

        self.output = self.layers[-1].activations

    def train(self, trainingset, maxit=10000, lrate=0.5, lambd=1000.0):

        for it in range(maxit):

            for t in trainingset:
                x = t[0]
                y = t[1]

                self.activate(x)
                out = self.output

                cost = 0
                for i in range(len(trainingset)):
                    for k in range(len(y)):
                        cost = cost + y[k] * log(out[k]) + \
                            (1 - y[k]) * log(1 - out[k])
                cost = -(1.0 / len(trainingset)) * cost

                for i in range(len(self.layers[-1].nodes)):
                    n = self.layers[-1].nodes[i]
                    a = n.activation
                    n.error = (y[i] - a) * dsigmoid(a)

                for l in range(len(self.layers) - 2, -1, -1):
                    fn = self.layers[l].nodes
                    for i in range(len(fn)):
                        tn = self.layers[l + 1].nodes

                        s = 0
                        for j in range(len(tn)):
                            w = tn[j].weights[i]
                            e = tn[j].error
                            s = s + e * w
                        a = fn[i].activation
                        fn[i].error = dsigmoid(a) * s

                for l in range(len(self.layers) - 1, -1, -1):
                    for i in range(len(self.layers[l].nodes)):
                        n = self.layers[l].nodes[i]

                        for j in range(len(n.weights)):
                            w = n.weights[j]
                            n.weights[j] = w + lrate * n.error * n.inputs[j]
                        n.bias_weight = n.bias_weight + lrate * n.error
            if it % 100 == 0:
                print cost

            if cost <= 1.0E-2:
                break

import json

# hidden = Layer()
# hidden.nodes = [Node(), Node()]

# output = Layer()
# output.nodes = [Node()]

# nn = NeuralNetwork()
# nn.set_input_size(2)
# nn.add_layer(hidden)
# nn.add_layer(output)

trainingset = [
    ([0, 0], [1]),
    ([1, 0], [0]),
    ([0, 1], [0]),
    ([1, 1], [1])
]

# nn.train(trainingset, 1000000)

# for t in trainingset:
#     nn.activate(t[0])
#     print nn.output

# out = []
# for l in nn.layers:
#     nw = []
#     for n in range(len(l.nodes)):
#         nw.append([l.nodes[n].weights, \
#             l.nodes[n].bias_weight])
#     out.append(nw)

# with open('dump.nn', 'w') as f:
#     f.write(json.dumps(out))

nnd = ''
with open('dump.nn', 'r') as f:
    nnd = json.loads(f.read())

nn = NeuralNetwork()
for l in nnd:
    layer = Layer()
    for n in l:
        node = Node()
        node.weights = n[0]
        node.bias_weight = n[1]
        layer.nodes.append(node)
    nn.add_layer(layer, False)

for t in trainingset:
    nn.activate(t[0])
    print nn.output