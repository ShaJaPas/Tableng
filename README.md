# Tableng
## Overview

This project contains of:

* Parser, for parsing languge constructions into AST
* Bytecode compiler, which compiles AST to bytecode
* Register-based VM for executing bytecode

## Code example
```rust
// Function defined in rust interpreter
extern fn print(x: any)

// Recursive fibonacci
fn fib(n: int) -> int {
    if n < 2 {
        return n
    } else {
        return fib(n - 2) + fib(n - 1)
    }
}

// Function to calculate Collatz conjecture sequence length
fn calc(index: int) -> int {
    var step = 0
    while index > 1 {
        if index % 2 == 0 {
            index = index / 2
            step = step + 1
        } else {
            index = (3 * index + 1) / 2
            step = step + 2
        }
    }

    return step
}

var result = 0

// Find the longest sequence in [1, 1000000]
for x in 1..1000000 {
    var e = calc(x)
    if e > result {
        result = e
    }
}

print(result)

// Calc 30 fibonacci number
print(fib(30))

// All complex data types are tables
var t = {1, 2, 3}

t[0] = {2: "2"}

t[1] = fib

print(t)
```
