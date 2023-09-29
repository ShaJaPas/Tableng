use hashbrown::HashMap;
use ordered_float::OrderedFloat;
use radix_fmt::radix;

use crate::{
    bytecode::{Bytecode, Literal, Object},
    opcode::OpCode,
    ParseError,
};

const NEXT_INDEX: usize = 0;
const EQ_INDEX: usize = 1;
const NEQ_INDEX: usize = 2;
const GT_INDEX: usize = 3;
const GTE_INDEX: usize = 4;
const LT_INDEX: usize = 5;
const LTE_INDEX: usize = 6;

pub trait StackVM {
    type Stack;
    type OpCode;

    fn next(&mut self) -> Self::OpCode;
    fn pop(&mut self) -> Self::Stack;
    fn push(&mut self, node: Self::Stack);
}

#[derive(Copy, Clone, Debug)]
struct Frame {
    /// Index of the current instruction
    ip: usize,
    /// Pointer to the stack than the function call starts
    sp: usize,
}

impl Frame {
    #[inline(always)]
    fn new(ip: usize, sp: usize) -> Self {
        Frame { ip, sp }
    }
}

#[derive(Debug)]
pub struct VM {
    stack: Vec<Object>,
    bytecode: Bytecode,
    call_stack: Vec<Frame>,
    variables: Vec<usize>,
    ip: usize,
    loop_delta: usize,
}

impl Default for VM {
    fn default() -> Self {
        Self {
            stack: Vec::with_capacity(64),
            bytecode: Bytecode::empty(),
            ip: 0,
            call_stack: Vec::with_capacity(32),
            variables: Vec::new(),
            loop_delta: 0,
        }
    }
}
impl VM {
    #[inline(always)]
    fn jump(&mut self, ip: usize) {
        self.ip = ip;
    }

    #[inline(always)]
    fn peek(&self) -> &Object {
        self.stack.last().unwrap()
    }

    fn ext_print(&mut self) -> Object {
        let val = self.ext_to_string();
        self.push(val);
        let arg = self.pop();
        match arg {
            Object::Literal(Literal::Str(v)) => println!("{v}"),
            _ => unreachable!(),
        }
        Object::Nil
    }

    fn ext_to_string(&mut self) -> Object {
        fn _to_string(obj: &Object) -> String {
            match obj {
                Object::Literal(v) => v.to_string(),
                Object::Table(t) => {
                    let params = t
                        .iter()
                        .map(|(k, v)| format!("{k}: {}", _to_string(v)))
                        .fold(String::new(), |acc, v| {
                            if acc.is_empty() {
                                v
                            } else {
                                acc + ", " + &v
                            }
                        });
                    format!("{{{params}}}")
                }
                Object::ObjectRef(addr) => format!("reference to 0x{}", radix(*addr, 16)),
                Object::Function(addr) => format!("function at 0x{}", radix(*addr, 16)),
                Object::Nil => "nil".to_string(),
            }
        }

        let val = &self.pop();
        let obj = self.get_value_by_ref(val);
        let str = _to_string(obj);
        Object::Literal(Literal::Str(Box::new(str)))
    }

    #[inline(always)]
    fn pop_frame(&mut self) {
        // Skipping bounds check does not add significant performance improvement
        let frame = self.call_stack.pop().unwrap();
        unsafe { self.stack.set_len(frame.sp + 1) }; // Alternative: `truncate`, but we can ignore dropping values
        self.ip = frame.ip;
    }

    #[inline(always)]
    fn get_value_by_ref<'a>(&'a self, reference: &'a Object) -> &Object {
        let mut reference = reference;
        while let Object::ObjectRef(sp) = reference {
            reference = &self.stack[*sp];
        }
        reference
    }

    #[inline(always)]
    fn push_frame(&mut self, ip: usize, sp: usize) {
        let frame = Frame::new(self.ip, sp);
        self.call_stack.push(frame);
        self.ip = ip; // Function start pointer
    }

    /// Clones const by index
    /// No bounds checking
    /// Performance: -1.2% over a regular `unwrap()`
    #[inline(always)]
    fn get_constant(&self, const_i: usize) -> Literal {
        unsafe { self.bytecode.constants.get_unchecked(const_i) }.clone()
    }

    #[inline(always)]
    fn get_local(&self, offset: usize) -> Object {
        let addr = self.call_stack.last().unwrap().sp + offset + 1;
        // Skipping bounds check does not add significant performance improvement (only 0.8%)
        let obj = &self.stack[addr];
        match obj {
            Object::Table(_) | Object::Function(_) => Object::ObjectRef(self.variables[offset]),
            _ => obj.clone(),
        }
    }

    #[inline(always)]
    fn call(&mut self, args_count: usize) -> Result<(), ParseError> {
        let func = &self.pop();
        let func = self.get_value_by_ref(func);
        if let Object::Function(addr) = func {
            self.push_frame(*addr, self.stack.len() - args_count - 1);
        } else {
            return Err(ParseError::RuntimeError {
                message: format!("Object `{:?}` is not callable", func),
            });
        };
        Ok(())
    }
    pub fn run(&mut self, code: Bytecode) -> Result<(), ParseError> {
        self.bytecode = code;
        self.ip = 0;
        self.stack.clear();
        self.call_stack.clear();
        self.call_stack.push(Frame::new(0, 0));
        self.variables.resize(self.bytecode.names.len(), 0);
        //println!("{:?}", self.bytecode.names);
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
                OpCode::Pop => {
                    self.pop();
                }
                OpCode::UnaryMinus => {
                    let rhs = self.pop();
                    match rhs {
                        Object::Literal(Literal::Int(v)) => {
                            self.push(Object::Literal(Literal::Int(-v)));
                        },
                        Object::Literal(Literal::Float(v)) => {
                            self.push(Object::Literal(Literal::Float(-v)));
                        },
                        _ => return Err(ParseError::RuntimeError { message: "Operation unsupported: Only integer of float values can allowed in `unary minus`".to_string() })
                    };
                }
                OpCode::UnaryNot => {
                    let rhs = self.pop();
                    match rhs {
                        Object::Literal(Literal::Bool(v)) => {
                            self.push(Object::Literal(Literal::Bool(!v)));
                        },
                        _ => return Err(ParseError::RuntimeError { message: "Operation unsupported: Only boolean values can allowed in `unary not`".to_string() })
                    };
                }
                OpCode::BinaryPlus => {
                    let rhs = self.pop();
                    let lhs = self.pop();
                    let res = match (lhs, rhs) {
                        (Object::Table(..), _) => unimplemented!(),
                        (Object::Literal(lhs), Object::Literal(rhs)) => match (lhs, rhs) {
                            (Literal::Int(lhs), Literal::Int(rhs)) => {
                                Object::Literal(Literal::Int(lhs + rhs))
                            }
                            (Literal::Float(lhs), Literal::Float(rhs)) => {
                                Object::Literal(Literal::Float(lhs + rhs))
                            }
                            (Literal::Int(lhs), Literal::Float(rhs)) => Object::Literal(
                                Literal::Float(OrderedFloat::from(lhs as f64) + rhs),
                            ),
                            (Literal::Float(lhs), Literal::Int(rhs)) => Object::Literal(
                                Literal::Float(lhs + OrderedFloat::from(rhs as f64)),
                            ),
                            _ => unreachable!(),
                        },
                        t => unreachable!("{t:?}"),
                    };
                    self.push(res);
                }
                OpCode::BinaryMinus => {
                    let rhs = self.pop();
                    let lhs = self.pop();
                    let res = match (lhs, rhs) {
                        (Object::Table(..), _) => unimplemented!(),
                        (Object::Literal(lhs), Object::Literal(rhs)) => match (lhs, rhs) {
                            (Literal::Int(lhs), Literal::Int(rhs)) => {
                                Object::Literal(Literal::Int(lhs - rhs))
                            }
                            (Literal::Float(lhs), Literal::Float(rhs)) => {
                                Object::Literal(Literal::Float(lhs - rhs))
                            }
                            (Literal::Int(lhs), Literal::Float(rhs)) => Object::Literal(
                                Literal::Float(OrderedFloat::from(lhs as f64) - rhs),
                            ),
                            (Literal::Float(lhs), Literal::Int(rhs)) => Object::Literal(
                                Literal::Float(lhs - OrderedFloat::from(rhs as f64)),
                            ),
                            _ => unreachable!(),
                        },
                        t => unreachable!("{t:?}"),
                    };
                    self.push(res);
                }
                OpCode::BinaryMultiplication => {
                    let rhs = self.pop();
                    let lhs = self.pop();
                    let res = match (lhs, rhs) {
                        (Object::Table(..), _) => unimplemented!(),
                        (Object::Literal(lhs), Object::Literal(rhs)) => match (lhs, rhs) {
                            (Literal::Int(lhs), Literal::Int(rhs)) => {
                                Object::Literal(Literal::Int(lhs * rhs))
                            }
                            (Literal::Float(lhs), Literal::Float(rhs)) => {
                                Object::Literal(Literal::Float(lhs * rhs))
                            }
                            (Literal::Int(lhs), Literal::Float(rhs)) => Object::Literal(
                                Literal::Float(OrderedFloat::from(lhs as f64) * rhs),
                            ),
                            (Literal::Float(lhs), Literal::Int(rhs)) => Object::Literal(
                                Literal::Float(lhs * OrderedFloat::from(rhs as f64)),
                            ),
                            _ => unreachable!(),
                        },
                        t => unreachable!("{t:?}"),
                    };
                    self.push(res);
                }
                OpCode::BinaryDivision => {
                    let rhs = self.pop();
                    let lhs = self.pop();
                    let res = match (lhs, rhs) {
                        (Object::Table(..), _) => unimplemented!(),
                        (Object::Literal(lhs), Object::Literal(rhs)) => match (lhs, rhs) {
                            (Literal::Int(lhs), Literal::Int(rhs)) => {
                                Object::Literal(Literal::Int(lhs / rhs))
                            }
                            (Literal::Float(lhs), Literal::Float(rhs)) => {
                                Object::Literal(Literal::Float(lhs / rhs))
                            }
                            (Literal::Int(lhs), Literal::Float(rhs)) => Object::Literal(
                                Literal::Float(OrderedFloat::from(lhs as f64) / rhs),
                            ),
                            (Literal::Float(lhs), Literal::Int(rhs)) => Object::Literal(
                                Literal::Float(lhs / OrderedFloat::from(rhs as f64)),
                            ),
                            _ => unreachable!(),
                        },
                        t => unreachable!("{t:?}"),
                    };
                    self.push(res);
                }
                OpCode::BinaryPower => {
                    let rhs = self.pop();
                    let lhs = self.pop();
                    let res = match (lhs, rhs) {
                        (Object::Literal(lhs), Object::Literal(rhs)) => match (lhs, rhs) {
                            (Literal::Int(lhs), Literal::Int(rhs)) => Object::Literal(
                                Literal::Float((lhs as f64).powf(rhs as f64).into()),
                            ),
                            (Literal::Float(lhs), Literal::Float(rhs)) => {
                                Object::Literal(Literal::Float(lhs.powf(*rhs).into()))
                            }
                            (Literal::Int(lhs), Literal::Float(rhs)) => {
                                Object::Literal(Literal::Float((lhs as f64).powf(*rhs).into()))
                            }
                            (Literal::Float(lhs), Literal::Int(rhs)) => {
                                Object::Literal(Literal::Float(lhs.powf(rhs as f64).into()))
                            }
                            _ => unreachable!(),
                        },
                        t => unreachable!("{t:?}"),
                    };
                    self.push(res);
                }
                OpCode::BinaryDiv => {
                    let rhs = self.pop();
                    let lhs = self.pop();
                    let res = match (lhs, rhs) {
                        (Object::Literal(lhs), Object::Literal(rhs)) => match (lhs, rhs) {
                            (Literal::Int(lhs), Literal::Int(rhs)) => {
                                Object::Literal(Literal::Int(lhs / rhs))
                            }
                            (Literal::Float(lhs), Literal::Float(rhs)) => {
                                Object::Literal(Literal::Float(lhs.div_euclid(*rhs).into()))
                            }
                            (Literal::Int(lhs), Literal::Float(rhs)) => {
                                Object::Literal(Literal::Float(
                                    OrderedFloat::from(lhs as f64).div_euclid(*rhs).into(),
                                ))
                            }
                            (Literal::Float(lhs), Literal::Int(rhs)) => {
                                Object::Literal(Literal::Float(
                                    lhs.div_euclid(*OrderedFloat::from(rhs as f64)).into(),
                                ))
                            }
                            _ => unreachable!(),
                        },
                        t => unreachable!("{t:?}"),
                    };
                    self.push(res);
                }
                OpCode::BinaryMod => {
                    let rhs = self.pop();
                    let lhs = self.pop();
                    let res = match (lhs, rhs) {
                        (Object::Literal(lhs), Object::Literal(rhs)) => match (lhs, rhs) {
                            (Literal::Int(lhs), Literal::Int(rhs)) => {
                                Object::Literal(Literal::Int(lhs % rhs))
                            }
                            (Literal::Float(lhs), Literal::Float(rhs)) => {
                                Object::Literal(Literal::Float(lhs % rhs))
                            }
                            (Literal::Int(lhs), Literal::Float(rhs)) => Object::Literal(
                                Literal::Float(OrderedFloat::from(lhs as f64) % rhs),
                            ),
                            (Literal::Float(lhs), Literal::Int(rhs)) => Object::Literal(
                                Literal::Float(lhs % OrderedFloat::from(rhs as f64)),
                            ),
                            _ => unreachable!(),
                        },
                        t => unreachable!("{t:?}"),
                    };
                    self.push(res);
                }
                OpCode::BinaryAnd => {
                    let rhs = self.pop();
                    let lhs = self.pop();
                    let res = match (lhs, rhs) {
                        (
                            Object::Literal(Literal::Bool(lhs)),
                            Object::Literal(Literal::Bool(rhs)),
                        ) => lhs && rhs,
                        (_, _) => {
                            return Err(ParseError::RuntimeError {
                                message: "Only booleans supports `binary and`".to_string(),
                            })
                        }
                    };
                    self.push(Object::Literal(Literal::Bool(res)));
                }
                OpCode::BinaryOr => {
                    let rhs = self.pop();
                    let lhs = self.pop();
                    let res = match (lhs, rhs) {
                        (
                            Object::Literal(Literal::Bool(lhs)),
                            Object::Literal(Literal::Bool(rhs)),
                        ) => lhs || rhs,
                        (_, _) => {
                            return Err(ParseError::RuntimeError {
                                message: "Only booleans supports `binary or`".to_string(),
                            })
                        }
                    };
                    self.push(Object::Literal(Literal::Bool(res)));
                }
                OpCode::BinaryEquals => {
                    let rhs_ref = &self.pop();
                    let lhs_ref = &self.pop();
                    let rhs = self.get_value_by_ref(rhs_ref);
                    let lhs = self.get_value_by_ref(lhs_ref);
                    let res = match (lhs, rhs) {
                        (Object::Table(inner), _) => {
                            let val = inner.get(&self.bytecode.constants[EQ_INDEX]);
                            if val.is_some() && matches!(val.unwrap(), Object::Function(_)) {
                                let func = val.cloned().unwrap();
                                self.push(rhs_ref.clone());
                                self.push(lhs_ref.clone());
                                self.push(func);
                                self.call(2)?;
                                self.pop()
                            } else {
                                return Err(ParseError::RuntimeError {
                                    message:
                                        "Table must have `__eq` method to perform `==` operation"
                                            .to_string(),
                                });
                            }
                        }
                        (lhs, rhs) => Object::Literal(Literal::Bool(*lhs == *rhs)),
                    };
                    self.push(res);
                }
                OpCode::BinaryNotEquals => {
                    let rhs_ref = &self.pop();
                    let lhs_ref = &self.pop();
                    let rhs = self.get_value_by_ref(rhs_ref);
                    let lhs = self.get_value_by_ref(lhs_ref);
                    let res = match (lhs, rhs) {
                        (Object::Table(inner), _) => {
                            let val = inner.get(&self.bytecode.constants[NEQ_INDEX]);
                            if val.is_some() && matches!(val.unwrap(), Object::Function(_)) {
                                let func = val.cloned().unwrap();
                                self.push(rhs_ref.clone());
                                self.push(lhs_ref.clone());
                                self.push(func);
                                self.call(2)?;
                                self.pop()
                            } else {
                                return Err(ParseError::RuntimeError {
                                    message:
                                        "Table must have `__neq` method to perform `!=` operation"
                                            .to_string(),
                                });
                            }
                        }
                        (lhs, rhs) => Object::Literal(Literal::Bool(*lhs != *rhs)),
                    };
                    self.push(res);
                }
                OpCode::BinaryLessThan => {
                    let rhs_ref = &self.pop();
                    let lhs_ref = &self.pop();
                    let rhs = self.get_value_by_ref(rhs_ref);
                    let lhs = self.get_value_by_ref(lhs_ref);
                    let res = match (lhs, rhs) {
                        (Object::Table(inner), _) => {
                            let val = inner.get(&self.bytecode.constants[LT_INDEX]);
                            if val.is_some() && matches!(val.unwrap(), Object::Function(_)) {
                                let func = val.cloned().unwrap();
                                self.push(rhs_ref.clone());
                                self.push(lhs_ref.clone());
                                self.push(func);
                                self.call(2)?;
                                self.pop()
                            } else {
                                return Err(ParseError::RuntimeError {
                                    message:
                                        "Table must have `__lt` method to perform `<` operation"
                                            .to_string(),
                                });
                            }
                        }
                        (Object::Literal(lhs), Object::Literal(rhs)) => match (lhs, rhs) {
                            (Literal::Int(lhs), Literal::Int(rhs)) => {
                                Object::Literal(Literal::Bool(lhs < rhs))
                            }
                            (Literal::Float(lhs), Literal::Float(rhs)) => {
                                Object::Literal(Literal::Bool(lhs < rhs))
                            }
                            (Literal::Int(lhs), Literal::Float(rhs)) => Object::Literal(
                                Literal::Bool(OrderedFloat::from(*lhs as f64) < *rhs),
                            ),
                            (Literal::Float(lhs), Literal::Int(rhs)) => Object::Literal(
                                Literal::Bool(*lhs < OrderedFloat::from(*rhs as f64)),
                            ),
                            _ => unreachable!(),
                        },
                        _ => unreachable!(),
                    };
                    self.push(res);
                }
                OpCode::BinaryLessThanOrEquals => {
                    let rhs_ref = &self.pop();
                    let lhs_ref = &self.pop();
                    let rhs = self.get_value_by_ref(rhs_ref);
                    let lhs = self.get_value_by_ref(lhs_ref);
                    let res = match (lhs, rhs) {
                        (Object::Table(inner), _) => {
                            let val = inner.get(&self.bytecode.constants[LTE_INDEX]);
                            if val.is_some() && matches!(val.unwrap(), Object::Function(_)) {
                                let func = val.cloned().unwrap();
                                self.push(rhs_ref.clone());
                                self.push(lhs_ref.clone());
                                self.push(func);
                                self.call(2)?;
                                self.pop()
                            } else {
                                return Err(ParseError::RuntimeError {
                                    message:
                                        "Table must have `__lte` method to perform `<=` operation"
                                            .to_string(),
                                });
                            }
                        }
                        (Object::Literal(lhs), Object::Literal(rhs)) => match (lhs, rhs) {
                            (Literal::Int(lhs), Literal::Int(rhs)) => {
                                Object::Literal(Literal::Bool(lhs <= rhs))
                            }
                            (Literal::Float(lhs), Literal::Float(rhs)) => {
                                Object::Literal(Literal::Bool(lhs <= rhs))
                            }
                            (Literal::Int(lhs), Literal::Float(rhs)) => Object::Literal(
                                Literal::Bool(OrderedFloat::from(*lhs as f64) <= *rhs),
                            ),
                            (Literal::Float(lhs), Literal::Int(rhs)) => Object::Literal(
                                Literal::Bool(*lhs <= OrderedFloat::from(*rhs as f64)),
                            ),
                            _ => unreachable!(),
                        },
                        _ => unreachable!(),
                    };
                    self.push(res);
                }
                OpCode::BinaryGreaterThan => {
                    let rhs_ref = &self.pop();
                    let lhs_ref = &self.pop();
                    let rhs = self.get_value_by_ref(rhs_ref);
                    let lhs = self.get_value_by_ref(lhs_ref);
                    let res = match (lhs, rhs) {
                        (Object::Table(inner), _) => {
                            let val = inner.get(&self.bytecode.constants[GT_INDEX]);
                            if val.is_some() && matches!(val.unwrap(), Object::Function(_)) {
                                let func = val.cloned().unwrap();
                                self.push(rhs_ref.clone());
                                self.push(lhs_ref.clone());
                                self.push(func);
                                self.call(2)?;
                                self.pop()
                            } else {
                                return Err(ParseError::RuntimeError {
                                    message:
                                        "Table must have `__gt` method to perform `>` operation"
                                            .to_string(),
                                });
                            }
                        }
                        (Object::Literal(lhs), Object::Literal(rhs)) => match (lhs, rhs) {
                            (Literal::Int(lhs), Literal::Int(rhs)) => {
                                Object::Literal(Literal::Bool(lhs > rhs))
                            }
                            (Literal::Float(lhs), Literal::Float(rhs)) => {
                                Object::Literal(Literal::Bool(lhs > rhs))
                            }
                            (Literal::Int(lhs), Literal::Float(rhs)) => Object::Literal(
                                Literal::Bool(OrderedFloat::from(*lhs as f64) > *rhs),
                            ),
                            (Literal::Float(lhs), Literal::Int(rhs)) => Object::Literal(
                                Literal::Bool(*lhs > OrderedFloat::from(*rhs as f64)),
                            ),
                            _ => unreachable!(),
                        },
                        _ => unreachable!(),
                    };
                    self.push(res);
                }
                OpCode::BinaryGreaterThanOrEquals => {
                    let rhs_ref = &self.pop();
                    let lhs_ref = &self.pop();
                    let rhs = self.get_value_by_ref(rhs_ref);
                    let lhs = self.get_value_by_ref(lhs_ref);
                    let res = match (lhs, rhs) {
                        (Object::Table(inner), _) => {
                            let val = inner.get(&self.bytecode.constants[GTE_INDEX]);
                            if val.is_some() && matches!(val.unwrap(), Object::Function(_)) {
                                let func = val.cloned().unwrap();
                                self.push(rhs_ref.clone());
                                self.push(lhs_ref.clone());
                                self.push(func);
                                self.call(2)?;
                                self.pop()
                            } else {
                                return Err(ParseError::RuntimeError {
                                    message:
                                        "Table must have `__gte` method to perform `>=` operation"
                                            .to_string(),
                                });
                            }
                        }
                        (Object::Literal(lhs), Object::Literal(rhs)) => match (lhs, rhs) {
                            (Literal::Int(lhs), Literal::Int(rhs)) => {
                                Object::Literal(Literal::Bool(lhs > rhs))
                            }
                            (Literal::Float(lhs), Literal::Float(rhs)) => {
                                Object::Literal(Literal::Bool(lhs > rhs))
                            }
                            (Literal::Int(lhs), Literal::Float(rhs)) => Object::Literal(
                                Literal::Bool(OrderedFloat::from(*lhs as f64) > *rhs),
                            ),
                            (Literal::Float(lhs), Literal::Int(rhs)) => Object::Literal(
                                Literal::Bool(*lhs > OrderedFloat::from(*rhs as f64)),
                            ),
                            _ => unreachable!(),
                        },
                        _ => unreachable!(),
                    };
                    self.push(res);
                }
                OpCode::Return => {
                    let res_ref = self.pop(); // Pop results or it will be erased
                    let res = self.get_value_by_ref(&res_ref).clone();
                    //println!("stack before {:?}", self.stack);
                    self.pop_frame();
                    let len = self.stack.len();
                    if self.loop_delta != 0 {
                        //self.stack.swap(len - 1, len - 2);
                        //println!("res: {:?}", res);
                        self.stack[len - 2] = res;
                        if matches!(self.stack[len - 2], Object::Nil) {
                            self.pop(); // pop iterator
                            self.pop(); // pop nil
                            self.jump(self.ip + self.loop_delta); // jump out of a loop
                        }
                    } else {
                        self.push(res); // Push it back
                    }
                    self.loop_delta = 0;
                    //println!("stack after {:?}", self.stack);
                }
                OpCode::Index => {
                    let index = self.pop();
                    let var = &self.pop();
                    let var = self.get_value_by_ref(var);
                    let res = match index {
                        Object::Literal(literal) => match var {
                            Object::Table(inner) => {
                                inner.get(&literal).unwrap_or(&Object::Nil).clone()
                            }
                            _ => {
                                return Err(ParseError::RuntimeError {
                                    message: "Indexes can be used only on tables".to_string(),
                                })
                            }
                        },
                        _ => {
                            return Err(ParseError::RuntimeError {
                                message: "Only literal values can be used as index".to_string(),
                            })
                        }
                    };
                    self.push(res);
                }
                OpCode::PushNil => self.push(Object::Nil),
                OpCode::StoreIndex => {
                    let index = self.pop();
                    let mut table = &self.pop();
                    let mut sp = self.stack.len() - 1;
                    let value = self.pop();

                    while let Object::ObjectRef(_sp) = table {
                        sp = *_sp;
                        table = &self.stack[sp];
                    }
                    let table = &mut self.stack[sp];
                    match index {
                        Object::Literal(literal) => match table {
                            Object::Table(inner) => {
                                inner.insert(literal, value);
                            }
                            _ => {
                                return Err(ParseError::RuntimeError {
                                    message: "Indexes can be used only on tables".to_string(),
                                })
                            }
                        },
                        _ => {
                            return Err(ParseError::RuntimeError {
                                message: "Only literal values can be used as index".to_string(),
                            })
                        }
                    };
                }
                OpCode::MakeVariable(var_i) => {
                    self.variables[var_i as usize] = self.stack.len() - 1;
                }
                OpCode::StoreVariable(var_i) => {
                    self.variables[var_i as usize] = self.stack.len() - 1;
                }
                OpCode::PushVariable(var_i) => {
                    let obj = &self.stack[self.variables[var_i as usize]];
                    let obj = match obj {
                        Object::Table(_) | Object::Function(_) => {
                            Object::ObjectRef(self.variables[var_i as usize])
                        }
                        _ => obj.clone(),
                    };
                    self.push(obj);
                }
                OpCode::PushConst(const_i) => {
                    self.push(Object::Literal(self.get_constant(const_i as usize)))
                }
                OpCode::MakeTable(len) => {
                    let mut table = HashMap::with_capacity(len as usize);
                    for _ in 0..len {
                        let key = self.pop();
                        let key = match key {
                            Object::Literal(literal) => literal,
                            _ => {
                                return Err(ParseError::RuntimeError {
                                    message: "Table keys must be literal types".to_string(),
                                })
                            }
                        };
                        let value = self.pop();
                        table.insert(key, value);
                    }
                    self.push(Object::Table(Box::new(table)));
                    self.push(Object::ObjectRef(self.stack.len() - 1));
                }
                OpCode::Call(args_count) => self.call(args_count as usize)?,
                OpCode::CallBuiltin(name_i) => {
                    match self.bytecode.extern_fn_names[name_i as usize].as_str() {
                        "print" => {
                            self.ext_print();
                        }
                        "to_string" => {
                            let val = self.ext_to_string();
                            self.push(val);
                        }
                        _ => unimplemented!(),
                    }
                }
                OpCode::JumpForward(delta) => self.jump(self.ip + delta as usize),
                OpCode::JumpBackward(delta) => self.jump(self.ip - delta as usize),
                OpCode::JumpForwardIfTrue(delta) => {
                    let flag = self.pop();
                    if let Object::Literal(Literal::Bool(flag)) = flag {
                        if flag {
                            self.jump(self.ip + delta as usize);
                        }
                    } else {
                        return Err(ParseError::RuntimeError {
                            message: "Type mismatch: Condition must be boolean type".to_string(),
                        });
                    }
                }
                OpCode::JumpBackwardIfTrue(delta) => {
                    let flag = self.pop();
                    if let Object::Literal(Literal::Bool(flag)) = flag {
                        if flag {
                            self.jump(self.ip - delta as usize);
                        }
                    } else {
                        return Err(ParseError::RuntimeError {
                            message: "Type mismatch: Condition must be boolean type".to_string(),
                        });
                    }
                }
                OpCode::JumpForwardIfFalse(delta) => {
                    let flag = self.pop();
                    if let Object::Literal(Literal::Bool(flag)) = flag {
                        if !flag {
                            self.jump(self.ip + delta as usize);
                        }
                    } else {
                        return Err(ParseError::RuntimeError {
                            message: "Type mismatch: Condition must be boolean type".to_string(),
                        });
                    }
                }
                OpCode::JumpBackwardIfFalse(delta) => {
                    let flag = self.pop();
                    if let Object::Literal(Literal::Bool(flag)) = flag {
                        if !flag {
                            self.jump(self.ip - delta as usize);
                        }
                    } else {
                        return Err(ParseError::RuntimeError {
                            message: "Type mismatch: Condition must be boolean type".to_string(),
                        });
                    }
                }
                OpCode::MakeFunction(addr) => {
                    self.push(Object::Function(addr as usize));
                }
                OpCode::GetNext(delta) => {
                    //println!("next: {:?}", self.stack);
                    self.loop_delta = delta as usize;
                    let iter_ref = self.peek();
                    let iterator = self.get_value_by_ref(iter_ref);
                    match iterator {
                        Object::Table(inner) => {
                            let next = inner.get(&self.bytecode.constants[NEXT_INDEX]);
                            if next.is_some() && matches!(next.unwrap(), Object::Function(_)) {
                                let next = next.unwrap().clone();
                                self.push(iter_ref.clone());
                                self.push(next);
                                self.call(1)?;
                            } else {
                                return Err(ParseError::RuntimeError {
                                    message: "Iterator table must implement `__next` method"
                                        .to_string(),
                                });
                            }
                        }
                        _ => {
                            return Err(ParseError::RuntimeError {
                                message: "Iterator object must be a table".to_string(),
                            })
                        }
                    }
                }
                OpCode::Nop => continue,
                OpCode::PushLocalVariable(offset) => {
                    self.push(self.get_local(offset as usize));
                }
                OpCode::StoreLocalVariable(offset) => {
                    let addr = self.call_stack.last().unwrap().sp + offset as usize + 1;
                    self.stack[addr] = self.pop();
                }
            }
        }

        println!("{:?}", self.stack);
        Ok(())
    }
}

impl StackVM for VM {
    type Stack = Object;
    type OpCode = OpCode;

    /// Pop an object off the stack
    /// This is like `Vec::pop`, but without checking if it's empty first.
    /// Performance: -22% over a regular call to `Vec::pop()`
    #[inline(always)]
    fn pop(&mut self) -> Self::Stack {
        // SAFETY: if the compiler and VM are implemented correctly, the stack will never be empty
        unsafe {
            let new_len = self.stack.len() - 1;
            self.stack.set_len(new_len);
            std::ptr::read(self.stack.as_ptr().add(new_len))
        }
    }

    #[inline(always)]
    fn push(&mut self, node: Self::Stack) {
        self.stack.push(node);
    }

    /// Copies next OpCde
    /// No bounds checking
    /// Performance: -1.2% over a regular `unwrap()`
    #[inline(always)]
    fn next(&mut self) -> Self::OpCode {
        // SAFETY: if the compiler and VM are implemented correctly, the ip will not go out of bounds
        let op_code = *unsafe { self.bytecode.instructions.get_unchecked(self.ip) };
        self.ip += 1;
        op_code
    }
}
