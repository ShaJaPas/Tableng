use hashbrown::HashMap;
use ordered_float::OrderedFloat;
use radix_fmt::radix;

use crate::{
    ir_gen::{Bytecode, Literal, Object},
    opcode_v2::OpCode,
    ParseError,
};

const NEXT_INDEX: usize = 0;
const EQ_INDEX: usize = 1;
const NEQ_INDEX: usize = 2;
const GT_INDEX: usize = 3;
const GTE_INDEX: usize = 4;
const LT_INDEX: usize = 5;
const LTE_INDEX: usize = 6;

#[derive(Copy, Clone, Debug)]
struct Frame {
    /// Index of the current instruction
    ip: usize,
    /// Pointer to the stack than the function call starts
    sp: usize,
    /// Number of virtual registers function uses
    vregs: usize,
}

impl Frame {
    #[inline(always)]
    fn new(ip: usize, sp: usize, vregs: usize) -> Self {
        Frame { ip, sp, vregs }
    }
}

#[derive(Debug)]
pub struct VM {
    heap: Vec<Object>,
    bytecode: Bytecode,
    call_stack: Vec<Frame>,
    stack: Vec<Object>,
    ip: usize,
    bp: usize,
}

impl Default for VM {
    fn default() -> Self {
        Self {
            heap: Vec::new(),
            bytecode: Bytecode::empty(),
            ip: 0,
            call_stack: Vec::with_capacity(32),
            stack: Vec::new(),
            bp: 0,
        }
    }
}
impl VM {
    //Builtin
    fn ext_print(&mut self, dst: usize) -> Object {
        let val = self.ext_to_string(dst);
        self.set_reg_value(dst, val);
        let arg = self.get_reg_value(dst);
        match arg {
            Object::Literal(Literal::Str(v)) => println!("{v}"),
            _ => unreachable!(),
        }
        Object::Nil
    }

    fn ext_to_string(&mut self, dst: usize) -> Object {
        fn _to_string(vm: &VM, obj: &Object) -> String {
            match obj {
                Object::Literal(v) => v.to_string(),
                Object::Table(t) => {
                    let params = t
                        .iter()
                        .map(|(k, v)| format!("{k}: {}", _to_string(vm, v)))
                        .fold(String::new(), |acc, v| {
                            if acc.is_empty() {
                                v
                            } else {
                                acc + ", " + &v
                            }
                        });
                    format!("{{{params}}}")
                }
                Object::ObjectRef(addr) => {
                    let rf = Object::ObjectRef(*addr);
                    let obj = vm.get_value_by_ref(&rf);
                    _to_string(vm, obj)
                },
                Object::Function(addr) => format!("function at 0x{}", radix(*addr, 16)),
                Object::Nil => "nil".to_string(),
            }
        }

        let val = &self.get_reg_value(dst);
        let obj = self.get_value_by_ref(val);
        let str = _to_string(self, obj);
        Object::Literal(Literal::Str(Box::new(str)))
    }

    #[inline(always)]
    fn jump(&mut self, ip: usize) {
        self.ip = ip;
    }

    #[inline(always)]
    fn push(&mut self, obj: Object) -> usize {
        self.heap.push(obj);
        self.heap.len() - 1
    }

    #[inline(always)]
    fn get_value_by_ref<'a>(&'a self, reference: &'a Object) -> &Object {
        let mut reference = reference;
        while let Object::ObjectRef(sp) = reference {
            reference = &self.heap[*sp];
        }
        reference
    }

    #[inline(always)]
    fn pop_frame(&mut self) {
        //println!("before: {:?}", self.stack);
        // Skipping bounds check does not add significant performance improvement
        let frame = self.call_stack.pop().unwrap();
        self.ip = frame.ip;
        let last = self.call_stack.last().unwrap();
        self.bp = last.sp;
        unsafe { self.stack.set_len(self.bp + last.vregs) }; // Alternative: `truncate`, but we can ignore dropping values

        //self.stack.resize(self.bp + frame.vregs, Object::Nil);
        //println!("after sp: {}", self.bp);
        //println!("after: {:?}", self.stack);
    }

    #[inline(always)]
    fn push_frame(&mut self, ip: usize, sp: usize, vregs: usize) {
        self.bp = sp;
        //println!("sp: {}", self.bp);
        let frame = Frame::new(self.ip, sp, vregs);
        self.call_stack.push(frame);
        self.ip = ip; // Function start pointer
    }

    /// Clones const by index
    /// No bounds checking
    /// Performance: -1.2% over a regular `unwrap()`
    #[inline(always)]
    fn get_constant(&self, const_i: usize) -> &Literal {
        unsafe { self.bytecode.constants.get_unchecked(const_i) }
    }

    #[inline(always)]
    fn get_reg_value(&self, reg: usize) -> &Object {
        unsafe { self.stack.get_unchecked(self.bp + reg) }
    }

    #[inline(always)]
    fn get_reg_value_mut(&mut self, reg: usize) -> &mut Object {
        unsafe { self.stack.get_unchecked_mut(self.bp + reg) }
    }

    #[inline(always)]
    fn set_reg_value(&mut self, reg: usize, obj: Object) {
        //println!("{obj:?}");
        //println!("{} {}", self.bp, reg);
        let idx = self.bp + reg;
        self.stack[idx] = obj;
    }

    /// Copies next OpCode
    /// No bounds checking
    /// Performance: -1.2% over a regular `unwrap()`
    #[inline(always)]
    fn next(&mut self) -> OpCode {
        // SAFETY: if the compiler and VM are implemented correctly, the ip will not go out of bounds
        let op_code = *unsafe { self.bytecode.instructions.get_unchecked(self.ip) };
        self.ip += 1;
        op_code
    }

    pub fn run(&mut self, code: Bytecode) -> Result<(), ParseError> {
        self.bytecode = code;
        self.ip = 0;
        self.heap.clear();
        self.call_stack.clear();
        self.call_stack
            .push(Frame::new(0, 0, self.bytecode.functions[0].0));
        self.stack = vec![Object::Nil; self.bytecode.functions[0].0];
        //println!("{:?}", self.bytecode.functions);
        println!("Bytecode (len)={:?}", self.bytecode.instructions.len());

        println!("Bytecode (raw)= \n{:?}", &self.bytecode.instructions);
        //println!("{:16}= {:?}", "Constants", self.bytecode.constants);

        //println!("{}", std::mem::size_of::<Object>());

        loop {
            //println!("{:16}= {:?}", "CallStack", self.call_stack);
            let next = self.next();
            //println!("{:16}= {:?}", "OpCode", next);
            //println!("{:16}={:?}", "Variables", self.variables.iter().map(|f| self.stack.get(*f)).collect::<Vec<Option<&Object>>>());
            //println!("{:?}", self.stack);
            //println!();

            match next {
                OpCode::Halt => break,
                OpCode::Nop => continue,
                OpCode::Jmp { delta } => self.jump((self.ip as isize + delta as isize) as usize),
                OpCode::LoadBool { dst, val } => {
                    self.set_reg_value(dst as usize, Object::Literal(Literal::Bool(val)));
                }
                OpCode::LoadInt { dst, val } => {
                    self.set_reg_value(dst as usize, Object::Literal(Literal::Int(val as i64)));
                }
                OpCode::LoadConst { dst, c_idx } => {
                    let res = self.get_constant(c_idx as usize).clone();
                    self.set_reg_value(dst as usize, Object::Literal(res));
                }
                OpCode::SetNil { dst } => {
                    self.set_reg_value(dst as usize, Object::Nil);
                }
                OpCode::Not { src, dst } => {
                    let rhs = self.get_reg_value(src as usize);
                    let res = match rhs {
                        Object::Literal(Literal::Bool(v)) => {
                            Object::Literal(Literal::Bool(!v))
                        },
                        _ => return Err(ParseError::RuntimeError { message: "Operation unsupported: Only boolean values can allowed in `unary not`".to_string() })
                    };
                    self.set_reg_value(dst as usize, res);
                }
                OpCode::UnaryMinus { src, dst } => {
                    let rhs = self.get_reg_value(src as usize);
                    let res = match rhs {
                        Object::Literal(Literal::Int(v)) => {
                            Object::Literal(Literal::Int(-v))
                        },
                        Object::Literal(Literal::Float(v)) => {
                            Object::Literal(Literal::Float(-v))
                        },
                        _ => return Err(ParseError::RuntimeError { message: "Operation unsupported: Only integer of float values can allowed in `unary minus`".to_string() })
                    };
                    self.set_reg_value(dst as usize, res);
                }
                OpCode::Move { src, dst } => {
                    let src_obj = self.get_reg_value(src as usize);
                    let res = match src_obj {
                        Object::Table(_) => Object::ObjectRef(src as usize),
                        val => val.clone(),
                    };
                    self.set_reg_value(dst as usize, res);
                }
                OpCode::AddInt { lhs, dst, val } => {
                    let lhs = self.get_reg_value(lhs as usize);
                    let res = match lhs {
                        Object::Table(..) => unimplemented!(),
                        Object::Literal(lhs) => match lhs {
                            Literal::Int(lhs) => Object::Literal(Literal::Int(lhs + val as i64)),
                            Literal::Float(lhs) => {
                                Object::Literal(Literal::Float(lhs + OrderedFloat(val as f64)))
                            }
                            _ => unreachable!(),
                        },
                        t => unreachable!("{t:?}"),
                    };
                    self.set_reg_value(dst as usize, res);
                }
                OpCode::AddConst { src, dst, c_idx } => {
                    let lhs = self.get_reg_value(src as usize);
                    let rhs = self.get_constant(c_idx as usize);
                    let res = match lhs {
                        Object::Table(..) => unimplemented!(),
                        Object::Literal(lhs) => match (lhs, rhs) {
                            (Literal::Int(lhs), Literal::Int(rhs)) => {
                                Object::Literal(Literal::Int(lhs + rhs))
                            }
                            (Literal::Float(lhs), Literal::Float(rhs)) => {
                                Object::Literal(Literal::Float(*lhs + *rhs))
                            }
                            (Literal::Int(lhs), Literal::Float(rhs)) => Object::Literal(
                                Literal::Float(OrderedFloat::from(*lhs as f64) + rhs),
                            ),
                            (Literal::Float(lhs), Literal::Int(rhs)) => Object::Literal(
                                Literal::Float(lhs + OrderedFloat::from(*rhs as f64)),
                            ),
                            _ => unreachable!(),
                        },
                        t => unreachable!("{t:?} stack: {:?}", self.stack),
                    };
                    self.set_reg_value(dst as usize, res);
                }
                OpCode::Add { lhs, rhs, dst } => {
                    let lhs = self.get_reg_value(lhs as usize);
                    let rhs = self.get_reg_value(rhs as usize);

                    let res = match (lhs, rhs) {
                        (Object::Table(..), _) => unimplemented!(),
                        (Object::Literal(lhs), Object::Literal(rhs)) => match (lhs, rhs) {
                            (Literal::Int(lhs), Literal::Int(rhs)) => {
                                Object::Literal(Literal::Int(*lhs + *rhs))
                            }
                            (Literal::Float(lhs), Literal::Float(rhs)) => {
                                Object::Literal(Literal::Float(*lhs + *rhs))
                            }
                            (Literal::Int(lhs), Literal::Float(rhs)) => Object::Literal(
                                Literal::Float(OrderedFloat::from(*lhs as f64) + rhs),
                            ),
                            (Literal::Float(lhs), Literal::Int(rhs)) => Object::Literal(
                                Literal::Float(lhs + OrderedFloat::from(*rhs as f64)),
                            ),
                            _ => unreachable!(),
                        },
                        t => unreachable!("{t:?}"),
                    };
                    self.set_reg_value(dst as usize, res);
                }
                OpCode::Sub { lhs, rhs, dst } => {
                    let lhs = self.get_reg_value(lhs as usize);
                    let rhs = self.get_reg_value(rhs as usize);

                    let res = match (lhs, rhs) {
                        (Object::Table(..), _) => unimplemented!(),
                        (Object::Literal(lhs), Object::Literal(rhs)) => match (lhs, rhs) {
                            (Literal::Int(lhs), Literal::Int(rhs)) => {
                                Object::Literal(Literal::Int(lhs - rhs))
                            }
                            (Literal::Float(lhs), Literal::Float(rhs)) => {
                                Object::Literal(Literal::Float(*lhs - *rhs))
                            }
                            (Literal::Int(lhs), Literal::Float(rhs)) => Object::Literal(
                                Literal::Float(OrderedFloat::from(*lhs as f64) - rhs),
                            ),
                            (Literal::Float(lhs), Literal::Int(rhs)) => Object::Literal(
                                Literal::Float(lhs - OrderedFloat::from(*rhs as f64)),
                            ),
                            _ => unreachable!(),
                        },
                        t => unreachable!("{t:?}"),
                    };
                    self.set_reg_value(dst as usize, res);
                }

                OpCode::MulInt { lhs, dst, val } => {
                    let lhs = self.get_reg_value(lhs as usize);
                    let res = match lhs {
                        Object::Table(..) => unimplemented!(),
                        Object::Literal(lhs) => match lhs {
                            Literal::Int(lhs) => Object::Literal(Literal::Int(lhs * val as i64)),
                            Literal::Float(lhs) => {
                                Object::Literal(Literal::Float(lhs * OrderedFloat(val as f64)))
                            }
                            _ => unreachable!(),
                        },
                        t => unreachable!("{t:?}"),
                    };
                    self.set_reg_value(dst as usize, res);
                }
                OpCode::MulConst { src, dst, c_idx } => {
                    let lhs = self.get_reg_value(src as usize);
                    let rhs = self.get_constant(c_idx as usize);
                    let res = match lhs {
                        Object::Table(..) => unimplemented!(),
                        Object::Literal(lhs) => match (lhs, rhs) {
                            (Literal::Int(lhs), Literal::Int(rhs)) => {
                                Object::Literal(Literal::Int(lhs * rhs))
                            }
                            (Literal::Float(lhs), Literal::Float(rhs)) => {
                                Object::Literal(Literal::Float(*lhs * *rhs))
                            }
                            (Literal::Int(lhs), Literal::Float(rhs)) => Object::Literal(
                                Literal::Float(OrderedFloat::from(*lhs as f64) * rhs),
                            ),
                            (Literal::Float(lhs), Literal::Int(rhs)) => Object::Literal(
                                Literal::Float(lhs * OrderedFloat::from(*rhs as f64)),
                            ),
                            _ => unreachable!(),
                        },
                        t => unreachable!("{t:?}"),
                    };
                    self.set_reg_value(dst as usize, res);
                }
                OpCode::Mul { lhs, rhs, dst } => {
                    let lhs = self.get_reg_value(lhs as usize);
                    let rhs = self.get_reg_value(rhs as usize);

                    let res = match (lhs, rhs) {
                        (Object::Table(..), _) => unimplemented!(),
                        (Object::Literal(lhs), Object::Literal(rhs)) => match (lhs, rhs) {
                            (Literal::Int(lhs), Literal::Int(rhs)) => {
                                Object::Literal(Literal::Int(lhs * rhs))
                            }
                            (Literal::Float(lhs), Literal::Float(rhs)) => {
                                Object::Literal(Literal::Float(*lhs * *rhs))
                            }
                            (Literal::Int(lhs), Literal::Float(rhs)) => Object::Literal(
                                Literal::Float(OrderedFloat::from(*lhs as f64) * rhs),
                            ),
                            (Literal::Float(lhs), Literal::Int(rhs)) => Object::Literal(
                                Literal::Float(lhs * OrderedFloat::from(*rhs as f64)),
                            ),
                            _ => unreachable!(),
                        },
                        t => unreachable!("{t:?}"),
                    };
                    self.set_reg_value(dst as usize, res);
                }

                OpCode::DivInt { lhs, dst, val } => {
                    let lhs = self.get_reg_value(lhs as usize);
                    let res = match lhs {
                        Object::Table(..) => unimplemented!(),
                        Object::Literal(lhs) => match lhs {
                            Literal::Int(lhs) => Object::Literal(Literal::Int(lhs / val as i64)),
                            Literal::Float(lhs) => {
                                Object::Literal(Literal::Float(lhs / OrderedFloat(val as f64)))
                            }
                            _ => unreachable!(),
                        },
                        t => unreachable!("{t:?}"),
                    };
                    self.set_reg_value(dst as usize, res);
                }
                OpCode::DivConst { src, dst, c_idx } => {
                    let lhs = self.get_reg_value(src as usize);
                    let rhs = self.get_constant(c_idx as usize);
                    let res = match lhs {
                        Object::Table(..) => unimplemented!(),
                        Object::Literal(lhs) => match (lhs, rhs) {
                            (Literal::Int(lhs), Literal::Int(rhs)) => {
                                Object::Literal(Literal::Int(lhs / rhs))
                            }
                            (Literal::Float(lhs), Literal::Float(rhs)) => {
                                Object::Literal(Literal::Float(*lhs / *rhs))
                            }
                            (Literal::Int(lhs), Literal::Float(rhs)) => Object::Literal(
                                Literal::Float(OrderedFloat::from(*lhs as f64) / rhs),
                            ),
                            (Literal::Float(lhs), Literal::Int(rhs)) => Object::Literal(
                                Literal::Float(lhs / OrderedFloat::from(*rhs as f64)),
                            ),
                            _ => unreachable!(),
                        },
                        t => unreachable!("{t:?}"),
                    };
                    self.set_reg_value(dst as usize, res);
                }
                OpCode::Div { lhs, rhs, dst } => {
                    let lhs = self.get_reg_value(lhs as usize);
                    let rhs = self.get_reg_value(rhs as usize);

                    let res = match (lhs, rhs) {
                        (Object::Table(..), _) => unimplemented!(),
                        (Object::Literal(lhs), Object::Literal(rhs)) => match (lhs, rhs) {
                            (Literal::Int(lhs), Literal::Int(rhs)) => {
                                Object::Literal(Literal::Int(lhs / rhs))
                            }
                            (Literal::Float(lhs), Literal::Float(rhs)) => {
                                Object::Literal(Literal::Float(*lhs / *rhs))
                            }
                            (Literal::Int(lhs), Literal::Float(rhs)) => Object::Literal(
                                Literal::Float(OrderedFloat::from(*lhs as f64) / rhs),
                            ),
                            (Literal::Float(lhs), Literal::Int(rhs)) => Object::Literal(
                                Literal::Float(lhs / OrderedFloat::from(*rhs as f64)),
                            ),
                            _ => unreachable!(),
                        },
                        t => unreachable!("{t:?}"),
                    };
                    self.set_reg_value(dst as usize, res);
                }

                OpCode::EqInt { src, int, res } => {
                    let lhs = self.get_reg_value(src as usize);
                    let result = match lhs {
                        Object::Literal(l) => match l {
                            Literal::Int(lhs) => *lhs == int as i64,
                            Literal::Float(lhs) => OrderedFloat(int as f64) == *lhs,
                            _ => unimplemented!(),
                        },
                        _ => unimplemented!(),
                    };

                    if result != res {
                        self.jump(self.ip + 1);
                    }
                }
                OpCode::EqConst { src, c_idx, res } => {
                    let lhs = self.get_reg_value(src as usize);
                    let rhs = self.get_constant(c_idx as usize);
                    let result = match lhs {
                        Object::Literal(lhs) => lhs == rhs,
                        Object::Table(_) => unimplemented!(),
                        _ => false,
                    };

                    if result != res {
                        self.jump(self.ip + 1);
                    }
                }
                OpCode::Eq { lhs, rhs, res } => {
                    let lhs = self.get_reg_value(lhs as usize);
                    let rhs = self.get_reg_value(rhs as usize);
                    let result = match (rhs, lhs) {
                        (Object::Table(_), _) => unimplemented!(),
                        (lhs, rhs) => lhs == rhs,
                    };

                    if result != res {
                        self.jump(self.ip + 1);
                    }
                }
                OpCode::Lt { lhs, rhs, res } => {
                    let lhs = self.get_reg_value(lhs as usize);
                    let rhs = self.get_reg_value(rhs as usize);
                    let result = match (lhs, rhs) {
                        (Object::Table(_), _) => unimplemented!(),
                        (Object::Literal(lhs), Object::Literal(rhs)) => match (lhs, rhs) {
                            (Literal::Int(lhs), Literal::Int(rhs)) => lhs < rhs,
                            (Literal::Float(lhs), Literal::Float(rhs)) => lhs < rhs,
                            (Literal::Int(lhs), Literal::Float(rhs)) => {
                                OrderedFloat::from(*lhs as f64) < *rhs
                            }
                            (Literal::Float(lhs), Literal::Int(rhs)) => {
                                *lhs < OrderedFloat::from(*rhs as f64)
                            }
                            _ => {
                                return Err(ParseError::RuntimeError {
                                    message: "'>' can obly be applied to tables and numbers"
                                        .to_string(),
                                })
                            }
                        },
                        _ => unreachable!(),
                    };

                    if result != res {
                        self.jump(self.ip + 1);
                    }
                }
                OpCode::GtInt { src, int, res } => {
                    let src = self.get_reg_value(src as usize);
                    let result = match src {
                        Object::Literal(l) => match l {
                            Literal::Int(lhs) => *lhs > int as i64,
                            Literal::Float(lhs) => *lhs > OrderedFloat(int as f64),
                            _ => unimplemented!(),
                        },
                        _ => unimplemented!(),
                    };

                    if result != res {
                        self.jump(self.ip + 1);
                    }
                }
                OpCode::LtInt { src, int, res } => {
                    let src = self.get_reg_value(src as usize);
                    let result = match src {
                        Object::Literal(l) => match l {
                            Literal::Int(lhs) => *lhs < int as i64,
                            Literal::Float(lhs) => *lhs < OrderedFloat(int as f64),
                            _ => unimplemented!(),
                        },
                        t => unreachable!("{t:?}"),
                    };

                    if result != res {
                        self.jump(self.ip + 1);
                    }
                }
                OpCode::GteInt { src, int, res } => {
                    let src = self.get_reg_value(src as usize);
                    let result = match src {
                        Object::Literal(l) => match l {
                            Literal::Int(lhs) => *lhs >= int as i64,
                            Literal::Float(lhs) => *lhs >= OrderedFloat(int as f64),
                            _ => unimplemented!(),
                        },
                        _ => unimplemented!(),
                    };

                    if result != res {
                        self.jump(self.ip + 1);
                    }
                }
                OpCode::LtConst { src, res, c_idx } => {
                    let lhs = self.get_reg_value(src as usize);
                    let rhs = self.get_constant(c_idx as usize);
                    let result = match lhs {
                        Object::Literal(lhs) => match (lhs, rhs) {
                            (Literal::Int(lhs), Literal::Int(rhs)) => lhs < rhs,
                            (Literal::Float(lhs), Literal::Float(rhs)) => lhs < rhs,
                            (Literal::Int(lhs), Literal::Float(rhs)) => {
                                OrderedFloat::from(*lhs as f64) < *rhs
                            }
                            (Literal::Float(lhs), Literal::Int(rhs)) => {
                                *lhs < OrderedFloat::from(*rhs as f64)
                            }
                            _ => unreachable!(),
                        },
                        Object::Table(_) => unimplemented!(),
                        _ => false,
                    };

                    if result != res {
                        self.jump(self.ip + 1);
                    }
                }
                OpCode::Test { src, res } => {
                    let src = self.get_reg_value(src as usize);
                    let result = match src {
                        Object::Literal(Literal::Bool(v)) => *v,
                        _ => {
                            return Err(ParseError::RuntimeError {
                                message: "Only boolean values supports logical operations"
                                    .to_string(),
                            })
                        }
                    };

                    if result != res {
                        self.jump(self.ip + 1);
                    }
                }
                OpCode::NewTable { dst, len } => {
                    let table = Object::Table(Box::new(HashMap::with_capacity(len as usize)));
                    let pos = self.push(table);
                    self.set_reg_value(dst as usize, Object::ObjectRef(pos));
                }
                OpCode::SetIntConstIndex {
                    t_dst,
                    i_idx,
                    c_idx,
                } => {
                    let val = self.get_constant(c_idx as usize).clone();
                    match &self.get_reg_value(t_dst as usize) {
                        Object::ObjectRef(stack_pos) => {
                            let pos = *stack_pos;
                            match &mut self.heap[pos] {
                                Object::Table(inner) => {
                                    inner.insert(Literal::Int(i_idx as i64), Object::Literal(val));
                                }
                                _ => unreachable!(),
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                OpCode::SetIntIndex { t_dst, i_idx, val } => {
                    let val = self.get_reg_value(val as usize).clone();
                    match self.get_reg_value_mut(t_dst as usize) {
                        Object::ObjectRef(stack_pos) => {
                            let pos = *stack_pos;
                            match &mut self.heap[pos] {
                                Object::Table(inner) => {
                                    inner.insert(Literal::Int(i_idx as i64), val);
                                }
                                _ => unreachable!(),
                            }
                        }
                        _ => {
                            return Err(ParseError::RuntimeError {
                                message: "Only tables can be indexed".to_string(),
                            })
                        }
                    }
                }
                OpCode::SetConstIndex {
                    t_dst,
                    c_idx,
                    v_src,
                } => {
                    let l = self.get_constant(c_idx as usize).clone();
                    let val = self.get_reg_value(v_src as usize).clone();
                    match self.get_reg_value_mut(t_dst as usize) {
                        Object::ObjectRef(stack_pos) => {
                            let pos = *stack_pos;
                            match &mut self.heap[pos] {
                                Object::Table(inner) => {
                                    inner.insert(l, val);
                                }
                                _ => unreachable!(),
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                OpCode::SetIndexConstVal {
                    t_dst,
                    i_src,
                    v_idx,
                } => {
                    let val = self.get_constant(v_idx as usize).clone();
                    let key = match self.get_reg_value(i_src as usize) {
                        Object::Literal(key) => key.clone(),
                        _ => {
                            return Err(ParseError::RuntimeError {
                                message: "Only literal values can table keys".to_string(),
                            })
                        }
                    };
                    match self.get_reg_value_mut(t_dst as usize) {
                        Object::ObjectRef(stack_pos) => {
                            let pos = *stack_pos;
                            match &mut self.heap[pos] {
                                Object::Table(inner) => {
                                    inner.insert(key, Object::Literal(val));
                                }
                                _ => unreachable!(),
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                OpCode::SetIndex {
                    t_dst,
                    i_src,
                    v_src,
                } => {
                    let val = self.get_reg_value(v_src as usize).clone();
                    let key = match self.get_reg_value(i_src as usize) {
                        Object::Literal(key) => key.clone(),
                        _ => {
                            return Err(ParseError::RuntimeError {
                                message: "Only literal values can table keys".to_string(),
                            })
                        }
                    };
                    match self.get_reg_value_mut(t_dst as usize) {
                        Object::ObjectRef(stack_pos) => {
                            let pos = *stack_pos;
                            match &mut self.heap[pos] {
                                Object::Table(inner) => {
                                    inner.insert(key, val);
                                }
                                _ => unreachable!(),
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                OpCode::GetIntIndex { dst, t_src, int } => {
                    let res = match self.get_reg_value(t_src as usize) {
                        Object::ObjectRef(stack_pos) => {
                            let pos = *stack_pos;
                            match &self.heap[pos] {
                                Object::Table(inner) => inner
                                    .get(&Literal::Int(int as i64))
                                    .unwrap_or(&Object::Nil)
                                    .clone(),
                                _ => unreachable!(),
                            }
                        }
                        _ => unreachable!(),
                    };

                    self.set_reg_value(dst as usize, res);
                }
                OpCode::GetConstIndex { dst, t_src, v_idx } => {
                    let key = self.get_constant(v_idx as usize);
                    let res = match self.get_reg_value(t_src as usize) {
                        Object::ObjectRef(stack_pos) => {
                            let pos = *stack_pos;
                            match &self.heap[pos] {
                                Object::Table(inner) => {
                                    inner.get(key).unwrap_or(&Object::Nil).clone()
                                }
                                _ => unreachable!(),
                            }
                        }
                        _ => unreachable!(),
                    };

                    self.set_reg_value(dst as usize, res);
                }
                OpCode::GetIndex { dst, t_src, src } => {
                    let key = match self.get_reg_value(src as usize) {
                        Object::Literal(key) => key,
                        _ => {
                            return Err(ParseError::RuntimeError {
                                message: "Only literal values can table keys".to_string(),
                            })
                        }
                    };
                    let res = match self.get_reg_value(t_src as usize) {
                        Object::ObjectRef(stack_pos) => {
                            let pos = *stack_pos;
                            match &self.heap[pos] {
                                Object::Table(inner) => {
                                    inner.get(key).unwrap_or(&Object::Nil).clone()
                                }
                                _ => unreachable!(),
                            }
                        }
                        _ => unreachable!(),
                    };

                    self.set_reg_value(dst as usize, res);
                }
                OpCode::LoadFunction { dst, f_idx } => {
                    self.set_reg_value(dst as usize, Object::Function(f_idx as usize));
                }
                OpCode::Call { f_src } => {
                    let f_src = f_src as usize;
                    let func = self.get_reg_value(f_src);
                    match func {
                        Object::Function(f_idx) => {
                            let (vregs, addr) = self.bytecode.functions[*f_idx];
                            //println!("f:{:?}", self.stack);
                            self.stack.resize(f_src + self.bp + vregs + 1, Object::Nil);
                            self.push_frame(addr, f_src + self.bp + 1, vregs);
                            //println!("s: {:?}", self.stack);
                        }
                        _ => {
                            return Err(ParseError::RuntimeError {
                                message: "Object is not a function".to_string(),
                            })
                        }
                    }
                }
                OpCode::ReturnNil => {
                    self.stack[self.bp - 1] = Object::Nil;
                    self.pop_frame();
                }
                OpCode::ReturnConst { c_idx } => {
                    let c_idx = self.get_constant(c_idx as usize).clone();
                    self.stack[self.bp - 1] = Object::Literal(c_idx);
                    self.pop_frame();
                }
                OpCode::Mod { lhs, rhs, dst } => todo!(),
                OpCode::Pow { lhs, rhs, dst } => todo!(),
                OpCode::IntDiv { lhs, rhs, dst } => todo!(),
                OpCode::Return { src } => {
                    let dst_src = self.bp + src as usize;
                    //println!("[] = {:?}", self.stack);
                    self.stack.swap(self.bp - 1, dst_src);
                    //println!("[] = {:?}", self.stack);
                    //println!("ret {:?}", self.stack[self.bp - 1]);
                    self.pop_frame();
                }
                OpCode::Lte { lhs, rhs, res } => {
                    let lhs = self.get_reg_value(lhs as usize);
                    let rhs = self.get_reg_value(rhs as usize);
                    let result = match (lhs, rhs) {
                        (Object::Table(_), _) => unimplemented!(),
                        (Object::Literal(lhs), Object::Literal(rhs)) => match (lhs, rhs) {
                            (Literal::Int(lhs), Literal::Int(rhs)) => lhs <= rhs,
                            (Literal::Float(lhs), Literal::Float(rhs)) => lhs <= rhs,
                            (Literal::Int(lhs), Literal::Float(rhs)) => {
                                OrderedFloat::from(*lhs as f64) <= *rhs
                            }
                            (Literal::Float(lhs), Literal::Int(rhs)) => {
                                *lhs <= OrderedFloat::from(*rhs as f64)
                            }
                            _ => {
                                return Err(ParseError::RuntimeError {
                                    message: "'<=' can obly be applied to tables and numbers"
                                        .to_string(),
                                })
                            }
                        },
                        _ => unreachable!(),
                    };

                    if result != res {
                        self.jump(self.ip + 1);
                    }
                }
                OpCode::LteConst { src, c_idx, res } => {
                    let lhs = self.get_reg_value(src as usize);
                    let rhs = self.get_constant(c_idx as usize);
                    let result = match lhs {
                        Object::Literal(lhs) => match (lhs, rhs) {
                            (Literal::Int(lhs), Literal::Int(rhs)) => lhs <= rhs,
                            (Literal::Float(lhs), Literal::Float(rhs)) => lhs <= rhs,
                            (Literal::Int(lhs), Literal::Float(rhs)) => {
                                OrderedFloat::from(*lhs as f64) <= *rhs
                            }
                            (Literal::Float(lhs), Literal::Int(rhs)) => {
                                *lhs <= OrderedFloat::from(*rhs as f64)
                            }
                            _ => unreachable!(),
                        },
                        Object::Table(_) => unimplemented!(),
                        _ => false,
                    };

                    if result != res {
                        self.jump(self.ip + 1);
                    }
                }
                OpCode::GtConst { src, c_idx, res } => {
                    let lhs = self.get_reg_value(src as usize);
                    let rhs = self.get_constant(c_idx as usize);
                    let result = match lhs {
                        Object::Literal(lhs) => match (lhs, rhs) {
                            (Literal::Int(lhs), Literal::Int(rhs)) => lhs > rhs,
                            (Literal::Float(lhs), Literal::Float(rhs)) => lhs > rhs,
                            (Literal::Int(lhs), Literal::Float(rhs)) => {
                                OrderedFloat::from(*lhs as f64) > *rhs
                            }
                            (Literal::Float(lhs), Literal::Int(rhs)) => {
                                *lhs > OrderedFloat::from(*rhs as f64)
                            }
                            _ => unreachable!(),
                        },
                        Object::Table(_) => unimplemented!(),
                        _ => false,
                    };

                    if result != res {
                        self.jump(self.ip + 1);
                    }
                }
                OpCode::GteConst { src, c_idx, res } => {
                    let lhs = self.get_reg_value(src as usize);
                    let rhs = self.get_constant(c_idx as usize);
                    let result = match lhs {
                        Object::Literal(lhs) => match (lhs, rhs) {
                            (Literal::Int(lhs), Literal::Int(rhs)) => lhs >= rhs,
                            (Literal::Float(lhs), Literal::Float(rhs)) => lhs >= rhs,
                            (Literal::Int(lhs), Literal::Float(rhs)) => {
                                OrderedFloat::from(*lhs as f64) >= *rhs
                            }
                            (Literal::Float(lhs), Literal::Int(rhs)) => {
                                *lhs >= OrderedFloat::from(*rhs as f64)
                            }
                            _ => unreachable!(),
                        },
                        Object::Table(_) => unimplemented!(),
                        _ => false,
                    };

                    if result != res {
                        self.jump(self.ip + 1);
                    }
                }
                OpCode::LteInt { src, int, res } => {
                    let src = self.get_reg_value(src as usize);
                    let result = match src {
                        Object::Literal(l) => match l {
                            Literal::Int(lhs) => *lhs <= int as i64,
                            Literal::Float(lhs) => *lhs <= OrderedFloat(int as f64),
                            _ => unimplemented!(),
                        },
                        _ => unimplemented!(),
                    };

                    if result != res {
                        self.jump(self.ip + 1);
                    }
                }
                OpCode::CallBuiltin { f_idx, dst } => {
                    match self.bytecode.extern_fn_names[f_idx as usize].as_str() {
                        "print" => {
                            let res = self.ext_print(dst as usize + 1);
                            self.set_reg_value(dst as usize, res);
                        }
                        "to_string" => {
                            let res = self.ext_to_string(dst as usize + 1);
                            self.set_reg_value(dst as usize, res);
                        }
                        name => {
                            return Err(ParseError::RuntimeError {
                                message: format!("Extern function '{name}' not found"),
                            })
                        }
                    }
                }
                OpCode::ForLoop { src, dst } => {
                    let src = self.get_reg_value(src as usize);

                    let dst = dst as usize;
                    let table = self.get_value_by_ref(src);
                    match table {
                        Object::Table(inner) => {
                            let next = inner.get(&self.bytecode.constants[NEXT_INDEX]);
                            if next.is_some() && matches!(next.unwrap(), Object::Function(_)) {
                                match next.unwrap() {
                                    Object::Function(f_idx) => {
                                        let (vregs, addr) = self.bytecode.functions[*f_idx];
                                        self.set_reg_value(dst + 1, src.clone());
                                        self.stack.resize(dst + self.bp + vregs + 1, Object::Nil);
                                        self.push_frame(addr, dst + self.bp + 1, vregs);
                                    }
                                    _ => unreachable!(),
                                }
                            } else {
                                return Err(ParseError::RuntimeError {
                                    message: "Iterator table must implement `__next` method"
                                        .to_string(),
                                });
                            }
                        }
                        _ => {
                            return Err(ParseError::RuntimeError {
                                message: "Foreach loop expects only tables".to_string(),
                            })
                        }
                    }
                }
                OpCode::TestNil { src } => {
                    let src = self.get_reg_value(src as usize);
                    let result = !matches!(src, Object::Nil);

                    if result {
                        self.jump(self.ip + 1);
                    }
                }
                OpCode::ModConst { lhs, c_idx, dst } => {
                    let lhs = self.get_reg_value(lhs as usize);
                    let rhs = self.get_constant(c_idx as usize);
                    let res = match lhs {
                        Object::Table(..) => unimplemented!(),
                        Object::Literal(lhs) => match (lhs, rhs) {
                            (Literal::Int(lhs), Literal::Int(rhs)) => {
                                Object::Literal(Literal::Int(lhs % rhs))
                            }
                            (Literal::Float(lhs), Literal::Float(rhs)) => {
                                Object::Literal(Literal::Float(*lhs % *rhs))
                            }
                            (Literal::Int(lhs), Literal::Float(rhs)) => Object::Literal(
                                Literal::Float(OrderedFloat::from(*lhs as f64) % rhs),
                            ),
                            (Literal::Float(lhs), Literal::Int(rhs)) => Object::Literal(
                                Literal::Float(lhs % OrderedFloat::from(*rhs as f64)),
                            ),
                            _ => unreachable!(),
                        },
                        t => unreachable!("{t:?}"),
                    };
                    self.set_reg_value(dst as usize, res);
                }
            }
        }

        //println!("Stack: {:?}", self.stack);
        //println!("Heap: {:?}", self.heap);

        Ok(())
    }
}
