# 0. Overview of CasADi
CasADi is a framework for algorithmic (automatic) differentiation and numerical optimization, particularly suited for optimal control, nonlinear programming (NLP), and related applications. At its core, CasADi allows you to construct symbolic expressions (and functions) that can be differentiated and then evaluated numerically or handed off to nonlinear solvers (e.g., IPOPT, SNOPT).

One of the first things you’ll notice when you import CasADi in Python (or C++, MATLAB, etc.) is that there are three primary “matrix‐like” classes:

- SX: a symbolic type for sparse expressions, built as a lightweight directed‐acyclic‐graph (DAG) in C.

- MX: a more general symbolic type that represents expression graphs via pointers, supporting advanced graph manipulations (loops, conditionals, code generation).

- DM: a dense (numerical) matrix or vector type—you can think of it as “just numbers” (no further symbolic structure).

# 1. Why SX, MX, and DM?

- Symbolic vs Numeric:

    - Symbolic objects let CasADi know “this is an expression I might want to differentiate (or do other graph manipulations on),” whereas numeric objects (DM) are just leaf values.

    - Both SX and MX are “symbolic” in the sense that they form an expression graph; DM does not.

- SX (Sparse-DAG) vs MX (Matrix-Graph):

    - SX builds expression graphs as a simple DAG in plain C arrays. Every operation creates new nodes with integer indices; you get a statically defined graph. This is fast to build for small-to-medium expressions, but it doesn’t support more complex features (like graph slicing, loops, or runtime changes).

    - MX builds expression graphs using dynamic C++ pointers/objects, allowing you to manipulate subgraphs, generate C code, build (for example) NLPs with complicated indexing, or do conditional/loop constructs. It’s more flexible but a bit slower to assemble.

- DM (Dense Matrix):

    - Once you have a function (built out of SX or MX), you pass DM objects as arguments to get purely numeric evaluations (function values, Jacobians, Hessians, etc.). DM is, in effect, “just numbers,” stored in a dense array.

In practice, you’ll often:

- Choose SX or MX when you define symbolic variables and build up your expressions (objective, constraints, etc.).

- Create a CasADi Function (mapping from symbolic inputs → symbolic outputs).

- Call that Function with DM arguments (actual numeric data) to evaluate.

- Ask CasADi to provide derivatives (Jacobian, Hessian) either on SX or MX graphs—then compile or evaluate them numerically.


# 2. SX vs MX vs DM: Internal Representations and Trade-offs
## 2.1 SX (Sparse-DAG)

- Representation: Each SX object holds:

    - A small integer ID (node index).

    - A global “node pool” (arrays) storing operator types, child‐indices, etc.

    - Essentially, an SX graph is a static, singly‐rooted DAG.

-Key Properties:

    - Immutable DAG: Once you combine SX symbols, you get a DAG that cannot be reorganized.

    - Sparse focus: Suited for moderately sized expressions with decent sparsity (e.g., Jacobians where many entries are zero).

    - Fast construction: Creating and combining SX symbols is very fast for small programs.

    - Limitations:

        - No easy way to “slice” a subgraph or splice in new nodes at runtime.

        - Loops/“for” constructs can be unrolled but become tedious if you need dynamic indexing.

        - Code‐generation support is more limited.

## 2.2 MX (Matrix-Graph)

- Representation:

    - MX uses pointer‐based objects; each MX node is an object instance.

    - Allows dynamic graph manipulations: e.g., you can extract a subgraph, create loops (For/If in CasADi), and do advanced code generation.

- Key Properties:

    - Fully featured: Supports runtime resizing of graphs, slicing, advanced index manipulations, conditional operations, loops (with casadi.if_else, casadi.while_loop, casadi.for_loop).

    - Slower assembly: Because each node is a C++ object, creating/growing an MX graph is somewhat slower than SX for a comparable expression.

    - Better for large-scale problems: If you have big NLPs (e.g., 5,000 states, 10,000 constraints) or want to exploit structure, MX can handle partitioning, block-sparsity, etc.

    - Code generation: MX’s graph can be exported to C code, or you can generate ODE integrators, QP solvers, etc.

## 2.3 DM (Dense Matrix)

- Representation:

    - Strictly numeric, dense storage (like a NumPy array).

    - No symbolic nodes—just a container for doubles/floats.

- Use Cases:

    - Provide actual numerical inputs to a CasADi function (e.g., initial guesses, parameter values).

    - Receive outputs (function values, Jacobians, etc.) as a DM.