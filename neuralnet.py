from math import e, log
from random import uniform


def sigmoid(z):
    return 1 / (1 + e ** (-z))


def dsigmoid(a):
    return a * (1 - a)


def cost_equation(hx, y):
    return -y * log(hx) - (1 - y) * log(1 - hx)

layers = [2, 3, 1]
nl = len(layers)
lrate = 0.5

# l to from
# [{0:[0,1]}]

w = []

for l in range(nl - 1):
    w.append({})
    for t in range(layers[l + 1]):
        w[l][t] = []
        for f in range(layers[l]):
            w[l][t].append(uniform(-0.3, 0.3))
        w[l][t].append(uniform(-0.3, 0.3))


def activate(x, l):
    assert l > 0 and l < nl, "invalid layer"
    assert layers[l - 1] == len(x), "invalid input size"

    an = []
    for i in range(layers[l]):
        s = 0.0
        for j in range(layers[l - 1]):
            s = s + w[l - 1][i][j] * x[j]  # Wij * Xj -> i = to; j = from
        s = s + w[l - 1][i][layers[l - 1]]  # bias
        an.append(sigmoid(s))
    return an


def update_weights(f, t, a, e):
    from_num = layers[f]
    to_num = layers[t]
    for ti in range(to_num):
        for fi in range(from_num):
            w[f][ti][fi] = w[f][ti][fi] + lrate * e[ti] * a[fi]
        w[f][ti][from_num] = w[f][ti][from_num] + lrate * e[ti]


def compute_errors(l):
    e = []
    s = 0.0


def prop_forward(x):
    """ returns a list with the activation values by layer """

    a = [x]

    for l in range(nl - 1):
        a.append(activate(a[l], l + 1))

    return a


def prop_back():

    trainset = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0])
    ]
    cost = 1
    j = 0

    # iterate it until converge
    while(cost >= 1.0e-2):
        eo = 0.0
        cs = 0.0
        for i in range(len(trainset)):
            x = trainset[i][0]
            y = trainset[i][1]

            a = prop_forward(x)

            # compute the output errors
            eo = []
            for k in range(len(a[nl - 1])):
                eo.append((y[k] - a[nl - 1][k]) * dsigmoid(a[nl - 1][k]))
                # the cost sum
                cs = cs + cost_equation(a[nl - 1][k], y[k])

            # update the weights from hidden layer to output
            update_weights(1, 2, a[1], eo)

            # compute the hidden layers errors
            eh = []
            for l in range(nl - 2):
                el = []
                for k in range(len(a[l + 1])):
                    s = 0.0
                    for o in range(len(eo)):
                        s = s + w[l + 1][o][k] * eo[o] * dsigmoid(a[l + 1][k])
                    el.append(s)
                eh.append(el)

            # update the weights from input layer to hidden
            for l in range(nl - 2):
                update_weights(l, l + 1, a[l], eh[l])

        cost = (1.0 / len(trainset)) * cs

        if j % 10000 == 0:
            print cost
        j = j + 1

    print prop_forward(trainset[0][0])
    print prop_forward(trainset[1][0])
    print prop_forward(trainset[2][0])
    print prop_forward(trainset[3][0])

prop_back()


def test_forward():
    w[0][0][0] = 20
    w[0][0][1] = 20
    w[0][0][2] = -30

    w[0][1][0] = -20
    w[0][1][1] = -20
    w[0][1][2] = 10

    w[1][0][0] = 20
    w[1][0][1] = 20
    w[1][0][2] = -10

    testset = [
        ([0, 0], [1]),
        ([0, 1], [0]),
        ([1, 0], [0]),
        ([1, 1], [1])
    ]

    for ts in testset:
        print prop_forward(ts[0])
