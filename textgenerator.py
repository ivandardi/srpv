import string
import itertools
import random

def plates():

    letters = map(''.join, itertools.product(string.ascii_uppercase, repeat=3))

    numbers = ['{0:04d}'.format(i) for i in range(10000)]

    for l in letters:
        for n in numbers:
            yield l + n


with open('por.training_text.txt', 'w') as f:
    p = list(plates())
    for plate in random.sample(p, 1000):
        print(plate, file=f)

