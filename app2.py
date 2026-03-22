# app2.py


import random

def random_number():
    num = random.randint(1, 50)
    print("Random number is:", num)
    return num

def check_even_odd(n):
    if n % 2 == 0:
        print("Number is even")
    else:
        print("Number is odd")

if __name__ == "__main__":
    print("Running app2 file...")
    
    value = random_number()
    check_even_odd(value)

    print("Program finished")