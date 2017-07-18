from tf import *

trainers = {
    'a': softmax_train,
    'b': sigmoid_5_layers_train,
    'c': relu_5_layers_train,
}

prompt = """
Please Input the trainer:
-------------------------   
a: softmax
b: sigmoid_5_layers
c: relu_5_layers_train
-------------------------
trainer: (a)
"""

if __name__ == '__main__':
    key = input(prompt)

    train = trainers.get(key, softmax_train)
    train()
