# app2.py
# simple activity file

import random

def generate():
    number = random.randint(1, 100)
    print("Generated:", number)
    return number

def check(n):
    if n % 2 == 0:
        print("Even number")
    else:
        print("Odd number")

if __name__ == "__main__":
    print("Program started")
    val = generate()
    check(val)
    print("Done")