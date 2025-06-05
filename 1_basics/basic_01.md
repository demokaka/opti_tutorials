# Code Casadi guildline: the basics

Import casadi with
```
from casadi import *
```

# Understand MX and SX underhood
## 1. The MX version
```
A_mx = MX.sym('A', 2, 2)
b_mx = MX.sym('b', 2)
z     = A_mx @ b_mx

f_mx = Function('f_mx', [A_mx, b_mx], [z])
print("MX :")
f_mx.disp(True)
```
MX’s “compiled” algorithm
```
f_mx:(i0[2x2],i1[2])->(o0[2]) MXFunction
Algorithm:
@0 = zeros(2x1)
@1 = input[0][0]
@2 = input[1][0]
@0 = mac(@1,@2,@0)
output[0][0] = @0
```
Explaination:

`f_mx:(i0[2x2],i1[2])->(o0[2]) MXFunction`

- i0[2x2] means the first input (input[0]) is a 2×2 matrix (A_mx).

- i1[2] means the second input (input[1]) is a 2-vector (b_mx).

- o0[2] means the single output is a 2-vector (z).

`@0 = zeros(2x1)`

It allocates a 2×1 zero‐vector and calls it temporary node @0. This will hold the result of A @ b as it accumulates.

`@1 = input[0][0]`

- input[0] is the 2×2 matrix A.

- [0] indexes into that 2×2 in column‐major order; in CasADi’s internal ordering, A[i,j] is stored at index 2*j + i. So:

- input[0][0] is A[0,0].

This line reads A[0,0] into node @1.

`@2 = input[1][0]`

- input[1] is the 2-vector b.

- input[1][0] is b[0].

This line reads b[0] into node @2.

`@0 = mac(@1,@2,@0)`

mac(x,y,z) stands for “multiply‐accumulate”: it does z += x*y, elementwise if needed. Here, since @0 is a 2×1 vector and @1,@2 are scalars, this is an invocation of a batched (vectorized) multiply‐accumulate that will handle all of the necessary scalar ops for the first column of the matrix‐vector product. In fact, in a 2×2·2×1 product, you can think of “mac” doing

@0[0] += A[0,0] * b[0]
@0[1] += A[1,0] * b[0]

in one shot. (Underneath, mac knows to treat @1 and @2 as “this column of A” vs “that entry of b” and accumulates into the 2×1 @0 appropriately.)

Because the code only shows one mac, you might wonder: what about the second column of A? The trick is that CasADi’s internal representation of mac(@1,@2,@0) here is overloaded to handle the entire column; in a larger matrix it would loop, but for 2×2 it's just two scalar multiplies and two scalar adds, all packed into one “mac” operator in the graph. In other words, this single mac node is doing both:
```
@0[0] ← @0[0] + A[0,0] * b[0]
@0[1] ← @0[1] + A[1,0] * b[0]
```
and then (in a second mac) it would do the same for [0,1]·b[1]. The printed version collapsed both operations into one node because of internal sparsity/fusion logic. In any case, MX is grouping the scalar multiplies+adds into fused “mac” nodes rather than listing each separate scalar operation.

output[0][0] = @0

    Finally, MX assigns the 2×1 vector in @0 to the function’s output o0. In other words,

        z[:] = @0[:]

        which is exactly the 2×1 result of A @ b.

Because MX knows “this is a 2×2 times 2×1,” it emits a single (batched) multiply‐accumulate. If you had done a 3×3 or 100×100, you’d still see a small number of “mac” nodes—MX fuses the entire matrix‐vector product into a handful of higher‐level nodes.

## 2. The SX version
```
A_sx = SX.sym('A', 2, 2)
b_sx = SX.sym('b', 2)
z_sx = A_sx @ b_sx

f_sx = Function('f_sx', [A_sx, b_sx], [z_sx])
print("SX :")
f_sx.disp(True)
```
SX’s “compiled” algorithm
```
f_sx:(i0[2x2],i1[2])->(o0[2]) SXFunction
Algorithm:
@0 = input[0][0];
@1 = input[1][0];
@0 = (@0*@1);
@2 = input[0][2];
@3 = input[1][1];
@2 = (@2*@3);
@0 = (@0+@2);
output[0][0] = @0;
@0 = input[0][1];
@0 = (@0*@1);
@1 = input[0][3];
@1 = (@1*@3);
@0 = (@0+@1);
output[0][1] = @0;
```

# Cách CasADi tạo mã C (hoặc Simulink C API) một cách hiệu quả bằng Common Subexpression Elimination (CSE)
## 1. Mục tiêu của CasADi khi tạo mã
CasADi cho phép bạn:

- Viết biểu thức toán học biểu diễn hệ thống, hàm mục tiêu, ràng buộc...

- Sau đó chuyển thành mã C/C++ tối ưu, với:

    - Common Subexpression Elimination (CSE)

    - Memory-efficient computation graph

    - Giảm thiểu số phép toán


