from casadi import *

x = MX.sym('x')
y = MX.sym('y')
expr = sin(x + y) + (x + y)

simplified_expr = cse(expr)  # Perform common subexpression elimination

f = Function('f', [x, y], [simplified_expr])
f.generate('f.c')
# This will generate a C code file named 'f.c' containing the function definition
# for the expression sin(x + y) + (x + y).