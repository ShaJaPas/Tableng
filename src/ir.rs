use regalloc2::PReg;
use strum::{EnumCount, FromRepr};

use crate::opcode_v2::OpCode;

#[derive(PartialEq, Debug, Clone, Copy, EnumCount, FromRepr)]
pub enum IROpCode {
    Halt, // OpCode for VM to stop running code
    Nop,
    ReturnNil,
    DefParam {
        dst: usize,
    },
    Jmp {
        delta: i16,
    },

    LoadInt {
        dst: usize,
        val: i16,
    },
    LoadBool {
        dst: usize,
        val: bool,
    },
    LoadConst {
        dst: usize,
        c_idx: usize, //todo: index c_idx: u16
    },

    LoadFunction {
        dst: usize,
        f_idx: u16,
    },

    Move {
        dst: usize,
        src: usize,
    },

    Add {
        lhs: usize,
        rhs: usize,
        dst: usize,
    },
    AddConst {
        src: usize,
        dst: usize,
        c_idx: u8,
    },
    AddInt {
        src: usize,
        dst: usize,
        int: i8,
    },

    Sub {
        lhs: usize,
        rhs: usize,
        dst: usize,
    },

    Mul {
        lhs: usize,
        rhs: usize,
        dst: usize,
    },
    MulConst {
        src: usize,
        dst: usize,
        c_idx: u8,
    },

    Div {
        lhs: usize,
        rhs: usize,
        dst: usize,
    },
    DivConst {
        src: usize,
        dst: usize,
        c_idx: u8,
    },

    Mod {
        lhs: usize,
        rhs: usize,
        dst: usize,
    },
    ModConst {
        lhs: usize,
        c_idx: u8,
        dst: usize,
    },
    Pow {
        lhs: usize,
        rhs: usize,
        dst: usize,
    },
    IntDiv {
        lhs: usize,
        rhs: usize,
        dst: usize,
    },
    Not {
        src: usize,
        dst: usize,
    },
    UnaryMinus {
        src: usize,
        dst: usize,
    },
    SetNil {
        dst: usize,
    },
    NewTable {
        dst: usize,
        len: usize,
    },
    SetIndex {
        t_dst: usize,
        i_src: usize,
        v_src: usize,
    },
    SetIndexConstVal {
        t_dst: usize,
        i_src: usize,
        v_idx: u8,
    },
    SetConstIndex {
        t_dst: usize,
        c_idx: u8,
        v_src: usize,
    },
    SetIntIndex {
        t_dst: usize,
        i_idx: i8,
        val: usize,
    },
    SetIntConstIndex {
        t_dst: usize,
        i_idx: i8,
        c_idx: u8,
    },
    GetIndex {
        dst: usize,
        t_src: usize,
        v_src: usize,
    },
    GetIntIndex {
        dst: usize,
        t_src: usize,
        int: i8,
    },
    GetConstIndex {
        dst: usize,
        t_src: usize,
        v_idx: u8,
    },
    Call {
        f_src: usize,
    },

    CallBuiltin {
        f_idx: usize,
        dst: usize,
    },

    ReturnConst {
        c_idx: u16,
    },
    Return {
        src: usize,
    },

    ForLoop {
        src: usize,
        dst: usize,
    },

    Test {
        src: usize,
        res: bool,
    }, // Tests if src register is boolean and equals to res. If the result is not equal to res, then skip next instruction

    TestNil {
        src: usize,
    }, // Tests if src register equals to mil. If the result is not equal to nil, then skip next instruction

    Eq {
        lhs: usize,
        rhs: usize,
        res: bool,
    }, // Compares lhs and rhs for equality. If the result is not equal to res, then skip next instruction

    Lt {
        lhs: usize,
        rhs: usize,
        res: bool,
    }, // Compares lhs and rhs for <. If the result is not equal to res, then skip next instruction

    Lte {
        lhs: usize,
        rhs: usize,
        res: bool,
    }, // Compares lhs and rhs for <=. If the result is not equal to res, then skip next instruction

    EqConst {
        src: usize,
        c_idx: u8,
        res: bool,
    }, // Compares src and const for equality. If the result is not equal to res, then skip next instruction
    LtConst {
        src: usize,
        c_idx: u8,
        res: bool,
    }, // Compares src and const for equality. If the result is not equal to res, then skip next instruction
    LteConst {
        src: usize,
        c_idx: u8,
        res: bool,
    }, // Compares src and const for equality. If the result is not equal to res, then skip next instruction
    GtConst {
        src: usize,
        c_idx: u8,
        res: bool,
    }, // Compares src and const for equality. If the result is not equal to res, then skip next instruction
    GteConst {
        src: usize,
        c_idx: u8,
        res: bool,
    }, // Compares src and const for equality. If the result is not equal to res, then skip next instruction

    EqInt {
        src: usize,
        int: i8,
        res: bool,
    }, // Compares src and const for equality. If the result is not equal to res, then skip next instruction

    LtInt {
        src: usize,
        int: i8,
        res: bool,
    }, // Compares src and const for equality. If the result is not equal to res, then skip next instruction

    LteInt {
        src: usize,
        int: i8,
        res: bool,
    }, // Compares src and const for equality. If the result is not equal to res, then skip next instruction

    GtInt {
        src: usize,
        int: i8,
        res: bool,
    }, // Compares src and const for equality. If the result is not equal to res, then skip next instruction

    GteInt {
        src: usize,
        int: i8,
        res: bool,
    }, // Compares src and const for equality. If the result is not equal to res, then skip next instruction
}

impl IROpCode {
    pub fn build_opcode(ir_code: Vec<Self>, allocs: Vec<PReg>) -> Vec<OpCode> {
        let mut p = 0;
        ir_code
            .into_iter()
            .filter_map(|code| match code {
                IROpCode::ReturnConst { c_idx } => Some(OpCode::ReturnConst { c_idx }),
                IROpCode::Halt => Some(OpCode::Halt),
                IROpCode::Nop => Some(OpCode::Nop),
                IROpCode::ReturnNil => Some(OpCode::ReturnNil),
                IROpCode::Jmp { delta } => Some(OpCode::Jmp { delta }),
                IROpCode::LoadInt { dst: _, val } => {
                    let res = OpCode::LoadInt {
                        dst: allocs[p].index() as u8,
                        val,
                    };
                    p += 1;
                    Some(res)
                }
                IROpCode::LoadBool { dst: _, val } => {
                    let res = OpCode::LoadBool {
                        dst: allocs[p].index() as u8,
                        val,
                    };
                    p += 1;
                    Some(res)
                }
                IROpCode::LoadConst { dst: _, c_idx } => {
                    let res = OpCode::LoadConst {
                        dst: allocs[p].index() as u8,
                        c_idx: c_idx as u16,
                    };
                    p += 1;
                    Some(res)
                }
                IROpCode::Move { .. } => {
                    let res = OpCode::Move {
                        dst: allocs[p].index() as u8,
                        src: allocs[p + 1].index() as u8,
                    };
                    p += 2;
                    Some(res)
                }
                IROpCode::Add { .. } => {
                    let res = OpCode::Add {
                        lhs: allocs[p].index() as u8,
                        rhs: allocs[p + 1].index() as u8,
                        dst: allocs[p + 2].index() as u8,
                    };
                    p += 3;
                    Some(res)
                }
                IROpCode::AddConst {
                    src: _,
                    dst: _,
                    c_idx,
                } => {
                    let res = OpCode::AddConst {
                        src: allocs[p].index() as u8,
                        dst: allocs[p + 1].index() as u8,
                        c_idx,
                    };
                    p += 2;
                    Some(res)
                }
                IROpCode::Sub { .. } => {
                    let res = OpCode::Sub {
                        lhs: allocs[p].index() as u8,
                        rhs: allocs[p + 1].index() as u8,
                        dst: allocs[p + 2].index() as u8,
                    };
                    p += 3;
                    Some(res)
                }
                IROpCode::Mul { .. } => {
                    let res = OpCode::Mul {
                        lhs: allocs[p].index() as u8,
                        rhs: allocs[p + 1].index() as u8,
                        dst: allocs[p + 2].index() as u8,
                    };
                    p += 3;
                    Some(res)
                }
                IROpCode::MulConst {
                    src: _,
                    dst: _,
                    c_idx,
                } => {
                    let res = OpCode::MulConst {
                        src: allocs[p].index() as u8,
                        dst: allocs[p + 1].index() as u8,
                        c_idx,
                    };
                    p += 2;
                    Some(res)
                }
                IROpCode::Div { .. } => {
                    let res = OpCode::Div {
                        lhs: allocs[p].index() as u8,
                        rhs: allocs[p + 1].index() as u8,
                        dst: allocs[p + 2].index() as u8,
                    };
                    p += 3;
                    Some(res)
                }
                IROpCode::DivConst {
                    src: _,
                    dst: _,
                    c_idx,
                } => {
                    let res = OpCode::DivConst {
                        src: allocs[p].index() as u8,
                        dst: allocs[p + 1].index() as u8,
                        c_idx,
                    };
                    p += 2;
                    Some(res)
                }
                IROpCode::Mod { .. } => {
                    let res = OpCode::Mod {
                        lhs: allocs[p].index() as u8,
                        rhs: allocs[p + 1].index() as u8,
                        dst: allocs[p + 2].index() as u8,
                    };
                    p += 3;
                    Some(res)
                }
                IROpCode::Pow { .. } => {
                    let res = OpCode::Pow {
                        lhs: allocs[p].index() as u8,
                        rhs: allocs[p + 1].index() as u8,
                        dst: allocs[p + 2].index() as u8,
                    };
                    p += 3;
                    Some(res)
                }
                IROpCode::IntDiv { .. } => {
                    let res = OpCode::IntDiv {
                        lhs: allocs[p].index() as u8,
                        rhs: allocs[p + 1].index() as u8,
                        dst: allocs[p + 2].index() as u8,
                    };
                    p += 3;
                    Some(res)
                }
                IROpCode::Not { .. } => {
                    let res = OpCode::Not {
                        src: allocs[p].index() as u8,
                        dst: allocs[p + 1].index() as u8,
                    };
                    p += 2;
                    Some(res)
                }
                IROpCode::UnaryMinus { .. } => {
                    let res = OpCode::UnaryMinus {
                        src: allocs[p].index() as u8,
                        dst: allocs[p + 1].index() as u8,
                    };
                    p += 2;
                    Some(res)
                }
                IROpCode::SetNil { .. } => {
                    let res = OpCode::SetNil {
                        dst: allocs[p].index() as u8,
                    };
                    p += 1;
                    Some(res)
                }
                IROpCode::NewTable { dst: _, len } => {
                    let res = OpCode::NewTable {
                        dst: allocs[p].index() as u8,
                        len: len as u16,
                    };
                    p += 1;
                    Some(res)
                }
                IROpCode::SetIndex { .. } => {
                    let res = OpCode::SetIndex {
                        t_dst: allocs[p].index() as u8,
                        i_src: allocs[p + 1].index() as u8,
                        v_src: allocs[p + 2].index() as u8,
                    };
                    p += 3;
                    Some(res)
                }
                IROpCode::GetIndex { .. } => {
                    let res = OpCode::GetIndex {
                        dst: allocs[p].index() as u8,
                        t_src: allocs[p + 1].index() as u8,
                        src: allocs[p + 2].index() as u8,
                    };
                    p += 3;
                    Some(res)
                }
                IROpCode::Call { .. } => {
                    let res = OpCode::Call {
                        f_src: allocs[p].index() as u8,
                    };
                    p += 1;
                    Some(res)
                }
                IROpCode::Return { .. } => {
                    let res = OpCode::Return {
                        src: allocs[p].index() as u8,
                    };
                    p += 1;
                    Some(res)
                }
                IROpCode::Test { src: _, res } => {
                    let res = OpCode::Test {
                        src: allocs[p].index() as u8,
                        res,
                    };
                    p += 1;
                    Some(res)
                }
                IROpCode::Eq {
                    lhs: _,
                    rhs: _,
                    res,
                } => {
                    let res = OpCode::Eq {
                        lhs: allocs[p].index() as u8,
                        rhs: allocs[p + 1].index() as u8,
                        res,
                    };
                    p += 2;
                    Some(res)
                }
                IROpCode::Lt {
                    lhs: _,
                    rhs: _,
                    res,
                } => {
                    let res = OpCode::Lt {
                        lhs: allocs[p].index() as u8,
                        rhs: allocs[p + 1].index() as u8,
                        res,
                    };
                    p += 2;
                    Some(res)
                }
                IROpCode::Lte {
                    lhs: _,
                    rhs: _,
                    res,
                } => {
                    let res = OpCode::Lte {
                        lhs: allocs[p].index() as u8,
                        rhs: allocs[p + 1].index() as u8,
                        res,
                    };
                    p += 2;
                    Some(res)
                }
                IROpCode::EqConst { src: _, c_idx, res } => {
                    let res = OpCode::EqConst {
                        src: allocs[p].index() as u8,
                        c_idx,
                        res,
                    };
                    p += 1;
                    Some(res)
                }
                IROpCode::LtConst { src: _, c_idx, res } => {
                    let res = OpCode::LtConst {
                        src: allocs[p].index() as u8,
                        c_idx,
                        res,
                    };
                    p += 1;
                    Some(res)
                }
                IROpCode::LteConst { src: _, c_idx, res } => {
                    let res = OpCode::LteConst {
                        src: allocs[p].index() as u8,
                        c_idx,
                        res,
                    };
                    p += 1;
                    Some(res)
                }
                IROpCode::GtConst { src: _, c_idx, res } => {
                    let res = OpCode::GtConst {
                        src: allocs[p].index() as u8,
                        c_idx,
                        res,
                    };
                    p += 1;
                    Some(res)
                }
                IROpCode::GteConst { src: _, c_idx, res } => {
                    let res = OpCode::GteConst {
                        src: allocs[p].index() as u8,
                        c_idx,
                        res,
                    };
                    p += 1;
                    Some(res)
                }
                IROpCode::EqInt { src: _, int, res } => {
                    let res = OpCode::EqInt {
                        src: allocs[p].index() as u8,
                        int,
                        res,
                    };
                    p += 1;
                    Some(res)
                }
                IROpCode::LtInt { src: _, int, res } => {
                    let res = OpCode::LtInt {
                        src: allocs[p].index() as u8,
                        int,
                        res,
                    };
                    p += 1;
                    Some(res)
                }
                IROpCode::LteInt { src: _, int, res } => {
                    let res = OpCode::LteInt {
                        src: allocs[p].index() as u8,
                        int,
                        res,
                    };
                    p += 1;
                    Some(res)
                }
                IROpCode::GtInt { src: _, int, res } => {
                    let res = OpCode::GtInt {
                        src: allocs[p].index() as u8,
                        int,
                        res,
                    };
                    p += 1;
                    Some(res)
                }
                IROpCode::GteInt { src: _, int, res } => {
                    let res = OpCode::GteInt {
                        src: allocs[p].index() as u8,
                        int,
                        res,
                    };
                    p += 1;
                    Some(res)
                }
                IROpCode::SetConstIndex {
                    t_dst: _,
                    c_idx,
                    v_src: _,
                } => {
                    let res = OpCode::SetConstIndex {
                        t_dst: allocs[p].index() as u8,
                        c_idx,
                        v_src: allocs[p + 1].index() as u8,
                    };
                    p += 2;
                    Some(res)
                }
                IROpCode::SetIntIndex {
                    t_dst: _,
                    i_idx,
                    val: _,
                } => {
                    let res = OpCode::SetIntIndex {
                        t_dst: allocs[p].index() as u8,
                        i_idx,
                        val: allocs[p + 1].index() as u8,
                    };
                    p += 2;
                    Some(res)
                }
                IROpCode::SetIntConstIndex {
                    t_dst: _,
                    i_idx,
                    c_idx,
                } => {
                    let res = OpCode::SetIntConstIndex {
                        t_dst: allocs[p].index() as u8,
                        i_idx,
                        c_idx,
                    };
                    p += 1;
                    Some(res)
                }
                IROpCode::SetIndexConstVal {
                    t_dst: _,
                    i_src: _,
                    v_idx,
                } => {
                    let res = OpCode::SetIndexConstVal {
                        t_dst: allocs[p].index() as u8,
                        i_src: allocs[p + 1].index() as u8,
                        v_idx,
                    };
                    p += 2;
                    Some(res)
                }
                IROpCode::GetIntIndex {
                    dst: _,
                    t_src: _,
                    int,
                } => {
                    let res = OpCode::GetIntIndex {
                        dst: allocs[p].index() as u8,
                        t_src: allocs[p + 1].index() as u8,
                        int,
                    };
                    p += 2;
                    Some(res)
                }
                IROpCode::GetConstIndex {
                    dst: _,
                    t_src: _,
                    v_idx,
                } => {
                    let res = OpCode::GetConstIndex {
                        dst: allocs[p].index() as u8,
                        t_src: allocs[p + 1].index() as u8,
                        v_idx,
                    };
                    p += 2;
                    Some(res)
                }
                IROpCode::LoadFunction { dst: _, f_idx } => {
                    let res = OpCode::LoadFunction {
                        dst: allocs[p].index() as u8,
                        f_idx,
                    };
                    p += 1;
                    Some(res)
                }
                IROpCode::AddInt {
                    src: _,
                    dst: _,
                    int,
                } => {
                    let res = OpCode::AddInt {
                        lhs: allocs[p].index() as u8,
                        dst: allocs[p + 1].index() as u8,
                        val: int,
                    };
                    p += 2;
                    Some(res)
                }
                IROpCode::DefParam { .. } => {
                    p += 1;
                    None
                }
                IROpCode::CallBuiltin { f_idx, .. } => {
                    let res = OpCode::CallBuiltin {
                        f_idx: f_idx as u16,
                        dst: allocs[p].index() as u8,
                    };
                    p += 1;
                    Some(res)
                }
                IROpCode::ForLoop { .. } => {
                    let res = OpCode::ForLoop {
                        src: allocs[p].index() as u8,
                        dst: allocs[p + 1].index() as u8,
                    };
                    p += 2;
                    Some(res)
                }
                IROpCode::TestNil { .. } => {
                    let res = OpCode::TestNil {
                        src: allocs[p].index() as u8,
                    };
                    p += 1;
                    Some(res)
                }
                IROpCode::ModConst {
                    lhs: _,
                    c_idx,
                    dst: _,
                } => {
                    let res = OpCode::ModConst {
                        lhs: allocs[p].index() as u8,
                        c_idx,
                        dst: allocs[p + 1].index() as u8,
                    };
                    p += 2;
                    Some(res)
                }
            })
            .collect()
    }
}
