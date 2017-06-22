import itertools
import string
from random import random


def all_plates():
    letters = map(''.join, itertools.product(string.ascii_uppercase, repeat=3))
    numbers = list('{0:04d}'.format(i) for i in range(10000))

    for l in letters:
        for n in numbers:
            yield l + n


with open('brplate.trained_text', 'w') as f:
    for plate in all_plates():
        if random() < 0.0001:
            print(plate, file=f)

