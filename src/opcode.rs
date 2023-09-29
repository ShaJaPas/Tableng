#[derive(PartialEq, Debug, Clone, Copy)]
pub enum OpCode {
    Halt,                      // OpCode for VM to stop running code
    Pop,                       // Pops top a stack
    UnaryMinus,                // POP = -POP
    UnaryNot,                  // POP = !POP
    BinaryPlus,                // POP = POP1 + POP
    BinaryMinus,               // POP = POP1 - POP
    BinaryMultiplication,      // POP = POP1 * POP
    BinaryDivision,            // POP = POP1 / POP
    BinaryPower,               // POP = POP1 ^ POP
    BinaryDiv,                 // POP = POP1 \ POP (integer division)
    BinaryMod,                 // POP = POP1 % POP (modulus division)
    BinaryAnd,                 // POP = POP1 and POP (logical and)
    BinaryOr,                  // POP = POP1 or POP (logical or)
    BinaryEquals,              // POP = POP1 == POP
    BinaryNotEquals,           // POP = POP1 == POP
    BinaryLessThan,            // POP = POP1 < POP
    BinaryLessThanOrEquals,    // POP = POP1 <= POP
    BinaryGreaterThan,         // POP = POP1 > POP
    BinaryGreaterThanOrEquals, // POP = POP1 >= POP
    Nop,                       // Does nothing. Used to be replaced by compiler.
    Return,                    // Returns top of a stack
    Index,                     // POP = POP1[POP]
    PushNil,                   // POP = nil
    StoreIndex,                // POP1[POP] = POP2 (Store value into an index)
    MakeVariable(u32),         // "NAME" = POP (Create a variable with a value of a stack)
    StoreVariable(u32),        // "NAME" = POP (Assosiates variable with a value of a stack)
    PushVariable(u32),         // POP = VALUES["NAME"] (Clones variable to the top a stack)
    PushLocalVariable(u32),    // POP = LOCAL_VALUES[offset]
    StoreLocalVariable(u32),   // LOCAL_VALUES[offset] = POP
    PushConst(u32),            // POP = constants[i] (Pushes constant to the stack),
    MakeTable(u32), // POP = table[POP(n):POP(n-1) .. POP2: POP1] (Creates table, poping n * 2 values from a stack)
    Call(u32),      // Calls POP = POP(POP1 .. POP(n-1), POP(n))
    CallBuiltin(u32), // Calls builting function POP = POP(POP1 .. POP(n-1), POP(n))
    JumpForward(u32), // Jumps forward for delta opcodes
    JumpBackward(u32), // Jumps backward for delta opcodes
    JumpForwardIfTrue(u32), // Jumps forward for delta opcodes if POP is true
    JumpBackwardIfTrue(u32), // Jumps backward for delta opcodes if POP is true
    JumpForwardIfFalse(u32), // Jumps forward for delta opcodes if POP is false
    JumpBackwardIfFalse(u32), // Jumps backward for delta opcodes if POP is false
    MakeFunction(u32), // POP = Function(F_ADDR) (Creates a function with an address and associates it with a name)
    GetNext(u32), //TOS is an iterator. Call its __next() method. If this yields a new value, push it on the stack (leaving the iterator below it). If the iterator indicates it is exhausted(nil), TOS is popped, and the byte code counter is incremented by delta.
}
