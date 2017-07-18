from tf import *

trainers = {
    'a': softmax_train
}

prompt = """
Please Input the trainer:
-------------------------   
a: softmax
-------------------------
trainer: (a)
"""

if __name__ == '__main__':
    key = input(prompt)

    train = trainers.get(key, softmax_train)
    train()
