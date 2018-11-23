import numpy as np
import math
from scipy.special import expit as sigmoid

def tanh(X):
    return math.tanh(X)

def arctan(X):
    return math.atan(X)

def rl(X):
    return math.log(1 + X.sigmoid, 10)

hidden = 100
input = 20
output = 600
images = 10
examples= 30

#random_int = np.random.randint(len(images))
#current_image = np.expand_dims(images[random_int], axis = 0)

G_W1 = np.random.uniform(-1,1,size = (input, hidden))
G_W2 = np.random.uniform(-1,1,size = (hidden+1, hidden))
G_W3 = np.random.uniform(-1,1,size = (hidden+1, hidden))
G_W4 = np.random.uniform(-1,1,size = (hidden+1, hidden))
G_W5 = np.random.uniform(-1,1,size = (hidden+1, hidden))
G_W6 = np.random.uniform(-1,1,size = (hidden+1, output))

Z = np.random.uniform(-1, 1, size = [examples, input])

G_b1 = np.ones([input, 1])


Gl1 = np.append(Z.dot(G_W1), G_b1, 1)
print(Gl1)
Gl1A = arctan(Gl1)
Gl2 = Gl1A.dot(G_W2) + G_b2
Gl2A = rl(Gl2)
Gl3 = Gl2A.dot(G_W3) + G_b3
Gl13A = arctan(Gl3)

Gl3 = Gl13A.dot(G_W4) + G_b4
G14A = rl(Gl4)
Gl5 = Gl4A.dot(G_W5) + G_b5
Gl5A = tanh(Gl5)
Gl6 = Gl5A.dot(G_W6) + G_b6
Gl6A = rl(Gl6)
Gl7 = Gl6A.dot(G_W7) + G_b7

outdata = math.log(Gl7, 10)





