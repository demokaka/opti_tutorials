from casadi import *
# This is a simple example of using CasADi to create a symbolic expression

x = SX.sym('x')  # Create a symbolic variable x
y = SX.sym('y')  # Create another symbolic variable y

# Create a symbolic expression z = x^2 + y^2
z = x**2 + y**2
# Create a function that takes x and y as inputs and returns z  
f = Function('f', [x, y], [z])
# Evaluate the function at x=1 and y=2
result = f(1, 2)
print("The result of the function f at x=1 and y=2 is:", result)  # Output: 5
# This code demonstrates how to create symbolic variables and functions in CasADi
# and how to evaluate them at specific values.
# The result of the function f at x=1 and y=2 is: 5

w = x + y  # Create another symbolic expression w = x + y
g = w*y

print(type(w))  # Check the type of w
# Output: <class 'casadi.casadi.SX'>
print(type(g))  # Check the type of g
# Output: <class 'casadi.casadi.SX'>



a = MX.sym('a') # Create a symbolic variable a using MX type
b = MX.sym('b') # Create another symbolic variable b using MX type
# Create a symbolic expression c = a + b
c = a + b
d = c * b

print(type(c))  # Check the type of c
# Output: <class 'casadi.casadi.MX'>
print(type(d))  # Check the type of d
# Output: <class 'casadi.casadi.MX'>
# This code demonstrates the use of MX type in CasADi, which is more efficient for large-scale problems.
# The MX type is used for more complex expressions and allows for more efficient computations.
# The code also shows how to create symbolic variables and expressions using both SX and MX types.
# The SX type is used for simple expressions, while the MX type is used for more complex expressions.