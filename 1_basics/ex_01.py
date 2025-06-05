# CasADi expressions graphs and Functions

from casadi import *

# 1. SX and MX
# 1.1. Create an express graph
x = MX.sym('x')  
y = MX.sym('y')

z = sin(x * y)
print(z)

# By using an inlining call to a Function, replace x by 2*x in this expression graph
# such that you obtain a print representation
# sin(((2.*x)*y))
f = Function('f', [x, y], [z], {"always_inline": True})
print(f(2 * x, y))  # This will print the expression graph with x replaced by 2*x

# 1.2. Perform the same subtitution now with substitute
print(substitute(z, x, 2*x))

# 1.3. Create an MX expression graph z for matrix multiplication as follows:
A_mx = MX.sym('A', 2, 2)
b_mx = MX.sym('b', 2)
z = A_mx @ b_mx


f_mx = Function('f_mx', [A_mx, b_mx], [z])

A_sx = SX.sym('A', 2, 2)
b_sx = SX.sym('b', 2)
z_sx = A_sx @ b_sx
f_sx = Function('f_sx', [A_sx, b_sx], [z_sx])

print("MX :")
f_mx.disp(True)
print("")
print("SX :")
f_sx.disp(True)
print("")

# SX variant is much longer than MX variant

# 1.4. What if you compare f_sx with f_mx.expand()?
print("MX expanded:")
f_mx.expand().disp(True)
print("")
print("SX expanded:")
f_sx.expand().disp(True)
print("")

# 'expanded' variant is the same as if you had started out with SX symbols

# 1.5. What happens when you call f_sx with MX symbols? And f_mx with SX symbols?
print(f_mx(A_mx, b_mx))  # ➔ mac(A,b,zeros(2x1))
print(f_sx(A_sx, b_sx))  # ➔ [((A_0*b_0)+(A_2*b_1)), ((A_1*b_0)+(A_3*b_1))]
# However, the following will also work due to the flexibility of CasADi
print(f_mx(A_sx, b_sx))  # ➔ [((A_0*b_0)+(A_2*b_1)), ((A_1*b_0)+(A_3*b_1))]
print(f_sx(A_mx, b_mx))  # ➔ f_sx(A, b){0}

# MX functions can accept SX inputs (they’re quietly converted to MX), and you get back MX expressions (possibly still fused as “mac” if it’s a matrix‐vector).

# SX functions can accept MX inputs, but CasADi cannot lower an MX node into the SX DAG, so it simply wraps the call in one MX function‐call node. 
# That is why you see f_sx(A, b){0} instead of a fully expanded SX DAG.

# This automatic promotion lets you mix SX and MX freely, 
# but whenever there’s an SX‐built function applied to MX symbols, 
# CasADi will treat it as a single opaque MX function‐call rather than 
# expanding all the underlying scalar operations.


# 1.6. We will compare SX versus MX for a linear solve expression: z=solve(A_mx,b_mx). As before
# create corresponding Functions f_mx and f_sx. Verify that you get the same numerical result
# when passing some numerical values to f_mx and f_sx.
# Pick for example:
A_val = [[1, 3], [2, 4]]
b_val = [1, 4]

A_mx = MX.sym('A', 2, 2)
b_mx = MX.sym('b', 2)
z_mx = solve(A_mx, b_mx)
f_mx = Function('f_mx', [A_mx, b_mx], [z_mx])

A_sx = SX.sym('A', 2, 2)
b_sx = SX.sym('b', 2)
z_sx = solve(A_sx, b_sx)
f_sx = Function('f_sx', [A_sx, b_sx], [z_sx])

print("MX solve result:", f_mx(DM(A_val), DM(b_val)))  
print("SX solve result:", f_sx(DM(A_val), DM(b_val)))


# 1.7. Verify that expanding f_mx yields an error. To have a bit of intuition of why, have a glance at
# the generated file f_mx.c you get after issuing f_mx.generate('f_mx.c'). The underlying
# algorithm of the linear solve is a QR decomposition obtained from Householder refections.
# Looking at the function body of casadi_house, you’ll notice some branching (if_else)
# conditions near the end. Since branching is not efficient for SX, expansion is disallowed.