use strum::{EnumCount, FromRepr};

#[derive(PartialEq, Debug, Clone, Copy, EnumCount, FromRepr)]
pub enum OpCode {
    Halt, // OpCode for VM to stop running code
    Nop,
    ReturnNil,
    Jmp { delta: i16 },

    LoadInt { dst: u8, val: i16 },
    LoadBool { dst: u8, val: bool },
    LoadConst { dst: u8, c_idx: u16 },
    LoadFunction { dst: u8, f_idx: u16 },
    Move { dst: u8, src: u8 },

    Add { lhs: u8, rhs: u8, dst: u8 },
    AddInt { lhs: u8, dst: u8, val: i8 },
    AddConst { src: u8, dst: u8, c_idx: u8 },

    Sub { lhs: u8, rhs: u8, dst: u8 },

    Mul { lhs: u8, rhs: u8, dst: u8 },
    MulInt { lhs: u8, dst: u8, val: i8 },
    MulConst { src: u8, dst: u8, c_idx: u8 },

    Div { lhs: u8, rhs: u8, dst: u8 },
    DivInt { lhs: u8, dst: u8, val: i8 },
    DivConst { src: u8, dst: u8, c_idx: u8 },

    Mod { lhs: u8, rhs: u8, dst: u8 },
    ModConst { lhs: u8, c_idx: u8, dst: u8 },

    Pow { lhs: u8, rhs: u8, dst: u8 },
    IntDiv { lhs: u8, rhs: u8, dst: u8 },
    Not { src: u8, dst: u8 },
    UnaryMinus { src: u8, dst: u8 },
    SetNil { dst: u8 },
    NewTable { dst: u8, len: u16 },
    SetIndex { t_dst: u8, i_src: u8, v_src: u8 },
    SetIndexConstVal { t_dst: u8, i_src: u8, v_idx: u8 },
    SetConstIndex { t_dst: u8, c_idx: u8, v_src: u8 },
    SetIntIndex { t_dst: u8, i_idx: i8, val: u8 },
    SetIntConstIndex { t_dst: u8, i_idx: i8, c_idx: u8 },
    GetIndex { dst: u8, t_src: u8, src: u8 },
    GetIntIndex { dst: u8, t_src: u8, int: i8 },
    GetConstIndex { dst: u8, t_src: u8, v_idx: u8 },

    Call { f_src: u8 },
    CallBuiltin { f_idx: u16, dst: u8 },
    Return { src: u8 },
    ReturnConst { c_idx: u16 },

    ForLoop { src: u8, dst: u8 }, // Calls "__next" method of src table, and places result in dst
    Test { src: u8, res: bool }, // Tests if src register is boolean and equals to res. If the result is not equal to res, then skip next instruction
    TestNil { src: u8 },
    Eq { lhs: u8, rhs: u8, res: bool }, // Compares lhs and rhs for equality. If the result is not equal to res, then skip next instruction

    Lt { lhs: u8, rhs: u8, res: bool }, // Compares lhs and rhs for <. If the result is not equal to res, then skip next instruction

    Lte { lhs: u8, rhs: u8, res: bool }, // Compares lhs and rhs for <=. If the result is not equal to res, then skip next instruction

    EqConst { src: u8, c_idx: u8, res: bool }, // Compares src and const for equality. If the result is not equal to res, then skip next instruction
    LtConst { src: u8, c_idx: u8, res: bool }, // Compares src and const for equality. If the result is not equal to res, then skip next instruction
    LteConst { src: u8, c_idx: u8, res: bool }, // Compares src and const for equality. If the result is not equal to res, then skip next instruction
    GtConst { src: u8, c_idx: u8, res: bool }, // Compares src and const for equality. If the result is not equal to res, then skip next instruction
    GteConst { src: u8, c_idx: u8, res: bool }, // Compares src and const for equality. If the result is not equal to res, then skip next instruction

    EqInt { src: u8, int: i8, res: bool }, // Compares src and const for equality. If the result is not equal to res, then skip next instruction

    LtInt { src: u8, int: i8, res: bool }, // Compares src and const for equality. If the result is not equal to res, then skip next instruction

    LteInt { src: u8, int: i8, res: bool }, // Compares src and const for equality. If the result is not equal to res, then skip next instruction

    GtInt { src: u8, int: i8, res: bool }, // Compares src and const for equality. If the result is not equal to res, then skip next instruction

    GteInt { src: u8, int: i8, res: bool }, // Compares src and const for equality. If the result is not equal to res, then skip next instruction
}
