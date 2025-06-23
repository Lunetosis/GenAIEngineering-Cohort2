import sys

def add(a,b):
    print(a+b)
    
if __name__ == "__main__":
    choice = sys.argv[1]
    num1 = float(sys.argv[2])
    num2 = float(sys.argv[3])
    
    if choice.upper() == 'ADD':
        add(num1, num2)
    else:
        print("Invalid input")