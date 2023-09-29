use std::ops::Range;

use hashbrown::HashMap;
use ordered_float::OrderedFloat;
use regalloc2::{Block, Operand, PReg, RegClass, VReg};

use crate::{
    ast::{
        self, AnonymousFunctionNode, AssignmentNode, ExpressionNode, ExternFunctionDeclarationNode,
        ForeachLoopNode, FunctionCallNode, FunctionDeclarationNode, IfStatementNode, Node,
        VariableDeclarationNode, WhileLoopNode,
    },
    ir::IROpCode,
    opcode_v2::OpCode,
    CompileRegister,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Literal {
    Int(i64),
    Float(OrderedFloat<f64>),
    Str(Box<String>),
    Bool(bool),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Object {
    Literal(Literal),
    Table(Box<HashMap<Literal, Object>>),
    ObjectRef(usize),
    Function(usize),
    Nil,
}

impl std::fmt::Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Literal::Int(v) => write!(f, "{v}"),
            Literal::Float(v) => write!(f, "{v}"),
            Literal::Str(v) => write!(f, "{v}"),
            Literal::Bool(v) => write!(f, "{v}"),
        }
    }
}

impl From<ast::Literal> for Literal {
    fn from(value: ast::Literal) -> Self {
        match value {
            ast::Literal::Int(v) => Self::Int(v),
            ast::Literal::Float(v) => Self::Float(v),
            ast::Literal::Str(v) => Self::Str(v.into()),
            ast::Literal::Bool(v) => Self::Bool(v),
        }
    }
}

#[derive(Debug)]
pub struct BytecodeBuilder {
    names: Vec<HashMap<String, usize>>,
    constants: HashMap<Literal, usize>,
    extern_functions: HashMap<String, usize>,
    loop_start: usize,
    base_pointer: usize,
    next_reg: usize,
    functions: HashMap<String, (usize, usize, usize)>, //desc: name: index, vregs_num, address
    fixed_regs: HashMap<usize, usize>,                 // instr_idx: fixed reg num
    max_reg: usize,
    switch: usize,
    args_offset: usize,
    pub reg_num: usize,
    pub instructions: Vec<IROpCode>,
    pub operands: Vec<Vec<Operand>>,
    pub blocks: Vec<BasicBlock>,
}

#[derive(Debug, Clone)]
enum BytecodeResult {
    Reg(usize),
    Const(Literal),
    And(Vec<BytecodeResult>),
    Or(Vec<BytecodeResult>),
}

#[derive(PartialEq, Debug, Clone)]
pub struct Bytecode {
    pub extern_fn_names: Vec<String>,
    pub constants: Vec<Literal>,
    pub instructions: Vec<OpCode>,
    pub functions: Vec<(usize, usize)>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct BasicBlock {
    pub instructions: Range<usize>,
    pub successors: Vec<Block>,   // Indices of successor blocks
    pub predecessors: Vec<Block>, // Indices of predecessors blocks
    pub params: Vec<VReg>,
}

impl Bytecode {
    pub fn empty() -> Self {
        Self {
            constants: Vec::new(),
            instructions: Vec::new(),
            extern_fn_names: Vec::new(),
            functions: Vec::new(),
        }
    }
}

impl Default for BytecodeBuilder {
    fn default() -> Self {
        Self {
            names: vec![HashMap::new()],
            loop_start: 0,
            extern_functions: HashMap::new(),
            constants: vec!["__next", "__eq", "__neq", "__gt", "__gte", "__lt", "__lte"]
                .into_iter()
                .enumerate()
                .map(|(i, f)| (Literal::Str(Box::new(f.to_string())), i))
                .collect(),
            instructions: Vec::new(),
            functions: HashMap::new(),
            fixed_regs: HashMap::new(),
            next_reg: 0,
            base_pointer: 0,
            reg_num: 0,
            max_reg: 0,
            switch: 1,
            args_offset: 0,
            blocks: Vec::new(),
            operands: Vec::new(),
        }
    }
}

impl BytecodeBuilder {
    fn get_next_vreg(&mut self) -> usize {
        let reg = self.next_reg - self.base_pointer;
        self.next_reg += 1;
        reg
    }

    fn add_instruction(&mut self, op_code: IROpCode) {
        self.instructions.push(op_code);
    }

    fn add_constant(&mut self, l: Literal) -> usize {
        if let Ok(v) = self.constants.try_insert(l.clone(), self.constants.len()) {
            return *v;
        }
        *self.constants.get(&l).unwrap()
    }

    fn add_block(&mut self, block_idx: usize) {
        if !self.blocks.is_empty() {
            let from = &mut self.blocks[block_idx];
            from.instructions.end = self.instructions.len();
            if from.instructions.is_empty() {
                self.blocks.remove(block_idx);
                return;
            }
            let block = BasicBlock {
                instructions: from.instructions.end..from.instructions.end,
                successors: vec![],
                params: vec![],
                predecessors: vec![Block(block_idx as u32)],
            };
            let new_idx = self.blocks.len();
            self.blocks[block_idx]
                .successors
                .push(Block(new_idx as u32));

            self.blocks.push(block);
        } else {
            let block = BasicBlock {
                instructions: 0..0,
                successors: vec![],
                params: vec![],
                predecessors: vec![],
            };
            self.blocks.push(block);
        }
    }

    fn end_block(&mut self) {
        assert!(!self.blocks.is_empty());

        if let Some(last) = self.blocks.last_mut() {
            last.instructions.end = self.instructions.len();
            last.successors.clear();
        }
    }

    fn eval_end_or(&mut self, res: BytecodeResult, dst_reg: Option<usize>) -> BytecodeResult {
        match res {
            BytecodeResult::And(vals) => {
                let len = self.instructions.len();
                for val in vals {
                    if let BytecodeResult::Reg(reg) = val {
                        self.add_instruction(IROpCode::Test {
                            src: reg,
                            res: false,
                        });
                        self.add_instruction(IROpCode::Jmp { delta: 0 });
                    } else {
                        self.eval_end_or(val, None);
                    }
                }
                for i in self.instructions[len..]
                    .iter()
                    .enumerate()
                    .filter(|(_, f)| matches!(f, IROpCode::Jmp { delta } if *delta == 0 ))
                    .map(|(i, _)| i)
                    .collect::<Vec<usize>>()
                {
                    self.instructions[len + i] = IROpCode::Jmp {
                        delta: (self.instructions.len() - len - i) as i16,
                    };
                }
                if let Some(dst) = dst_reg {
                    self.add_instruction(IROpCode::Jmp { delta: 2 });
                    self.add_instruction(IROpCode::LoadBool { dst, val: false });
                    self.add_instruction(IROpCode::Jmp { delta: 1 });
                    self.add_instruction(IROpCode::LoadBool { dst, val: true });
                }
                BytecodeResult::Reg(dst_reg.unwrap_or_else(|| self.get_next_vreg()))
            }
            BytecodeResult::Or(vals) => {
                let len = self.instructions.len();
                for val in vals {
                    if let BytecodeResult::Reg(reg) = val {
                        self.add_instruction(IROpCode::Test {
                            src: reg,
                            res: true,
                        });
                        self.add_instruction(IROpCode::Jmp { delta: 0 });
                    } else {
                        self.eval_end_or(val, None);
                    }
                }
                for i in self.instructions[len..]
                    .iter()
                    .enumerate()
                    .filter(|(_, f)| matches!(f, IROpCode::Jmp { delta } if *delta == 0 ))
                    .map(|(i, _)| i)
                    .collect::<Vec<usize>>()
                {
                    self.instructions[len + i] = IROpCode::Jmp {
                        delta: (self.instructions.len() - len - i - 1) as i16,
                    };
                }
                if let Some(dst) = dst_reg {
                    self.add_instruction(IROpCode::Jmp { delta: 2 });
                    self.add_instruction(IROpCode::LoadBool { dst, val: false });
                    self.add_instruction(IROpCode::Jmp { delta: 1 });
                    self.add_instruction(IROpCode::LoadBool { dst, val: true });
                }
                BytecodeResult::Reg(dst_reg.unwrap_or_else(|| self.get_next_vreg()))
            }
            BytecodeResult::Reg(src) => {
                let dst = dst_reg.unwrap_or_else(|| self.get_next_vreg());
                self.add_instruction(IROpCode::Move { dst, src });
                BytecodeResult::Reg(dst)
            }
            t => unreachable!("{t:?}"),
        }
    }

    fn add_result(&mut self, val: BytecodeResult, reg: usize) {
        match val {
            BytecodeResult::Reg(src) => self.add_instruction(IROpCode::Move { dst: reg, src }),
            BytecodeResult::Const(l) => match l {
                Literal::Bool(val) => self.add_instruction(IROpCode::LoadBool { dst: reg, val }),
                Literal::Int(v) => {
                    if let Ok(val) = TryInto::<i16>::try_into(v) {
                        self.add_instruction(IROpCode::LoadInt { dst: reg, val })
                    } else {
                        let c_idx = self.add_constant(l);

                        self.add_instruction(IROpCode::LoadConst { dst: reg, c_idx })
                    }
                }
                _ => {
                    let c_idx = self.add_constant(l);
                    self.add_instruction(IROpCode::LoadConst { dst: reg, c_idx })
                }
            },
            val => {
                self.eval_end_or(val, Some(reg));
            }
        }
    }

    pub fn interpret_node(&mut self, node: Node, block_idx: Option<usize>) {
        match node {
            Node::Program(statments) => {
                if statments.is_empty() {
                    return;
                }

                for stmt in statments {
                    self.interpret_node(stmt, block_idx);
                }
            }
            Node::VariableDeclaration(VariableDeclarationNode {
                datatype: _, // TODO: runtime table type checking
                identifier,
                initializer,
            }) => {
                self.switch = 1;
                let len = self.instructions.len();
                let reg = self.get_next_vreg();
                self.names
                    .last_mut()
                    .unwrap()
                    .insert(identifier.clone(), reg);
                let res = self.eval_node(*initializer, Some(reg), false);
                if let Some(IROpCode::Call { f_src, .. }) = self.instructions.last() {
                    self.names
                        .last_mut()
                        .unwrap()
                        .insert(identifier.clone(), *f_src);
                }
                if len == self.instructions.len()
                    || matches!(res, BytecodeResult::And(_) | BytecodeResult::Or(_))
                {
                    self.add_result(res, reg);
                }
            }
            Node::Assignment(AssignmentNode {
                identifier,
                expression,
            }) => {
                let ident = *identifier;
                self.switch = 1;

                match ident {
                    ExpressionNode::UnaryOperation(op, rhs) => {
                        let rhs = self.eval_node(*rhs, None, false);
                        match op {
                            ast::UnaryOperator::Index(idx) => {
                                let idx = self.eval_node(*idx, None, false);
                                let res = self.eval_node(*expression, None, false);

                                match rhs {
                                    BytecodeResult::Reg(register) => match (idx, res) {
                                        (BytecodeResult::Reg(k), res) => match res {
                                            BytecodeResult::Reg(v) => {
                                                self.add_instruction(IROpCode::SetIndex {
                                                    t_dst: register,
                                                    i_src: k,
                                                    v_src: v,
                                                });
                                            }
                                            BytecodeResult::Const(v) => {
                                                let v_idx = self.add_constant(v) as u8;
                                                self.add_instruction(IROpCode::SetIndexConstVal {
                                                    t_dst: register,
                                                    i_src: k,
                                                    v_idx,
                                                });
                                            }
                                            _ => unreachable!(),
                                        },
                                        (BytecodeResult::Const(k), res) => match k {
                                            Literal::Int(k) => {
                                                if let Ok(val) = TryInto::<i8>::try_into(k) {
                                                    match res {
                                                        BytecodeResult::Reg(reg) => {
                                                            self.add_instruction(
                                                                IROpCode::SetIntIndex {
                                                                    t_dst: register,
                                                                    i_idx: val,
                                                                    val: reg,
                                                                },
                                                            );
                                                        }
                                                        BytecodeResult::Const(l) => {
                                                            let c_idx = self.add_constant(l) as u8;
                                                            self.add_instruction(
                                                                IROpCode::SetIntConstIndex {
                                                                    t_dst: register,
                                                                    i_idx: val,
                                                                    c_idx,
                                                                },
                                                            );
                                                        }
                                                        _ => unreachable!(),
                                                    }
                                                } else {
                                                    match res {
                                                        BytecodeResult::Reg(reg) => {
                                                            let c_idx = self
                                                                .add_constant(Literal::Int(k))
                                                                as u8;
                                                            self.add_instruction(
                                                                IROpCode::SetConstIndex {
                                                                    t_dst: register,
                                                                    c_idx,
                                                                    v_src: reg,
                                                                },
                                                            );
                                                        }
                                                        BytecodeResult::Const(l) => {
                                                            let c_idx = self.add_constant(l);
                                                            let reg = self.get_next_vreg();
                                                            self.add_instruction(
                                                                IROpCode::LoadConst {
                                                                    dst: reg,
                                                                    c_idx,
                                                                },
                                                            );
                                                            let c_idx = self
                                                                .add_constant(Literal::Int(k))
                                                                as u8;
                                                            self.add_instruction(
                                                                IROpCode::SetConstIndex {
                                                                    t_dst: register,
                                                                    c_idx,
                                                                    v_src: reg,
                                                                },
                                                            );
                                                        }
                                                        _ => unreachable!(),
                                                    }
                                                }
                                            }
                                            v => match res {
                                                BytecodeResult::Reg(reg) => {
                                                    let c_idx = self.add_constant(v) as u8;
                                                    self.add_instruction(IROpCode::SetConstIndex {
                                                        t_dst: register,
                                                        c_idx,
                                                        v_src: reg,
                                                    });
                                                }
                                                BytecodeResult::Const(l) => {
                                                    let c_idx = self.add_constant(l);
                                                    let reg = self.get_next_vreg();
                                                    self.add_instruction(IROpCode::LoadConst {
                                                        dst: reg,
                                                        c_idx,
                                                    });
                                                    let c_idx = self.add_constant(v) as u8;
                                                    self.add_instruction(IROpCode::SetConstIndex {
                                                        t_dst: register,
                                                        c_idx,
                                                        v_src: reg,
                                                    });
                                                }
                                                _ => unreachable!(),
                                            },
                                        },
                                        _ => unreachable!(),
                                    },
                                    _ => unreachable!(),
                                }
                            }
                            _ => unreachable!(),
                        }
                    }
                    ExpressionNode::Identifier(name) => {
                        let reg = *self.names.iter().find_map(|f| f.get(&name)).unwrap();
                        let len = self.instructions.len();
                        let res = self.eval_node(*expression, Some(reg), false);
                        if len == self.instructions.len()
                            || matches!(res, BytecodeResult::And(_) | BytecodeResult::Or(_))
                        {
                            self.add_result(res, reg);
                        }
                    }
                    _ => unreachable!(),
                }
            }
            Node::StatementBlock(node) => {
                self.add_block(block_idx.unwrap_or(self.blocks.len().saturating_sub(1)));
                self.interpret_node(*node, block_idx);
            }
            Node::Expression(node) => {
                self.switch = 1;

                let len = self.instructions.len();
                let reg = self.get_next_vreg();

                let res = self.eval_node(node, Some(reg), false);
                if len == self.instructions.len()
                    || matches!(res, BytecodeResult::And(_) | BytecodeResult::Or(_))
                {
                    self.add_result(res, reg);
                }
            }
            Node::IfStatement(IfStatementNode {
                condition,
                if_block,
                else_block,
            }) => {
                self.switch = 1;

                self.eval_node(*condition, None, true);

                /*self.add_result(res, reg);
                self.add_instruction(IROpCode::Test {
                    src: reg,
                    res: false,
                });*/
                self.add_instruction(IROpCode::Jmp { delta: 1 });
                let len = self.instructions.len();
                self.add_instruction(IROpCode::Nop);

                self.interpret_node(*if_block, Some(block_idx.unwrap_or(self.blocks.len() - 1)));
                if let Some(else_block) = else_block {
                    let jmp_out = self.instructions.len();
                    self.add_instruction(IROpCode::Nop);
                    self.instructions[len] = IROpCode::Jmp {
                        delta: (self.instructions.len() - 1 - len) as i16,
                    };
                    self.interpret_node(
                        *else_block,
                        Some(block_idx.unwrap_or(self.blocks.len() - 1)),
                    );
                    self.instructions[jmp_out] = IROpCode::Jmp {
                        delta: (self.instructions.len() - 1 - jmp_out) as i16,
                    };
                } else {
                    self.instructions[len] = IROpCode::Jmp {
                        delta: (self.instructions.len() - 1 - len) as i16,
                    };
                }
            }
            Node::WhileStatement(WhileLoopNode { condition, body }) => {
                self.switch = 1;

                //let reg = self.get_next_vreg();

                self.loop_start = self.instructions.len() - 1;
                self.eval_node(*condition, None, true);

                /*self.add_result(res, reg);
                self.add_instruction(IROpCode::Test {
                    src: reg,
                    res: false,
                });*/
                self.add_instruction(IROpCode::Jmp { delta: 1 });
                let len = self.instructions.len();
                self.add_instruction(IROpCode::Nop);
                self.interpret_node(*body, block_idx);
                self.add_instruction(IROpCode::Jmp {
                    delta: (self.loop_start as i64 - self.instructions.len() as i64) as i16,
                });
                self.instructions[len] = IROpCode::Jmp {
                    delta: (self.instructions.len() - 1 - len) as i16,
                };

                // Replace all NOPs by jump forward as it is break statement
                for i in self.instructions[self.loop_start..]
                    .iter()
                    .enumerate()
                    .filter(|(_, f)| matches!(f, IROpCode::Nop))
                    .map(|(i, _)| i)
                    .collect::<Vec<usize>>()
                {
                    self.instructions[self.loop_start + i] = IROpCode::Jmp {
                        delta: (self.instructions.len() - self.loop_start - i + 1) as i16,
                    };
                }
            }
            Node::BreakStatement => self.add_instruction(IROpCode::Nop),
            Node::ContinueStatement => self.add_instruction(IROpCode::Jmp {
                delta: (self.loop_start as i64 - self.instructions.len() as i64) as i16,
            }),
            Node::FunctionDeclaration(FunctionDeclarationNode {
                return_type: _,
                identifier,
                parameters,
                body,
            }) => {
                let bp = self.base_pointer;
                self.base_pointer = self.next_reg;
                let params_len = parameters.len();
                self.args_offset += params_len;
                self.next_reg += params_len;

                let func_num = self.functions.len() + 1;
                assert!(!self.functions.contains_key(&identifier));

                self.functions.insert(identifier.clone(), (func_num, 0, 0));
                self.names.push(HashMap::new());

                self.add_block(block_idx.unwrap_or(self.blocks.len() - 1));

                for (idx, (param, _)) in parameters.into_iter().enumerate() {
                    self.names.last_mut().unwrap().insert(param, idx);
                    self.add_instruction(IROpCode::DefParam { dst: idx });
                }
                let len = self.instructions.len();
                self.add_instruction(IROpCode::Nop);
                self.interpret_node(*body, block_idx);
                self.add_instruction(IROpCode::ReturnNil);
                self.instructions[len] = IROpCode::Jmp {
                    delta: (self.instructions.len() - len - 1) as i16,
                };
                self.functions.insert(
                    identifier,
                    (
                        func_num,
                        self.next_reg - self.base_pointer,
                        len - self.args_offset + 1,
                    ),
                );

                self.reg_num = std::cmp::max(self.reg_num, self.next_reg - self.base_pointer);

                self.next_reg = self.base_pointer;
                self.base_pointer = bp;

                self.names.pop();
            }
            Node::ReturnStatement(expr) => {
                self.switch = 1;

                if let Some(expr) = expr {
                    let res = self.eval_node(*expr, None, false);
                    match res {
                        BytecodeResult::Reg(reg) => {
                            self.add_instruction(IROpCode::Return { src: reg });
                        }
                        BytecodeResult::Const(l) => {
                            let c_idx = self.add_constant(l) as u16;
                            self.add_instruction(IROpCode::ReturnConst { c_idx });
                        }
                        _ => unreachable!(),
                    }
                } else {
                    self.add_instruction(IROpCode::ReturnNil);
                }
            }
            Node::ForeachStatement(ForeachLoopNode {
                identifier,
                table,
                body,
            }) => {
                self.add_block(block_idx.unwrap_or(self.blocks.len() - 1));
                self.names.push(HashMap::new());

                let src = self.get_next_vreg();
                self.eval_node(*table, Some(src), false);

                let dst = self.get_next_vreg();
                self.names.last_mut().unwrap().insert(identifier, dst);

                self.add_instruction(IROpCode::ForLoop { src, dst });
                self.add_instruction(IROpCode::TestNil { src: dst });

                let len = self.instructions.len();
                self.add_instruction(IROpCode::Nop);
                self.interpret_node(*body, None);
                self.add_instruction(IROpCode::Jmp {
                    delta: (len as i64 - 3 - self.instructions.len() as i64) as i16,
                });
                self.instructions[len] = IROpCode::Jmp {
                    delta: (self.instructions.len() - len - 1) as i16,
                };

                self.names.pop();
            }
            Node::ExternFunctionDeclaration(ExternFunctionDeclarationNode {
                return_type: _,
                identifier,
                parameters: _,
            }) => {
                self.extern_functions
                    .insert(identifier, self.extern_functions.len());
            }
        }
    }

    fn eval_node(
        &mut self,
        node: ExpressionNode,
        dst_reg: Option<usize>,
        condition: bool,
    ) -> BytecodeResult {
        macro_rules! load_bool_result {
            ($dst:expr) => {
                if !condition {
                    self.add_instruction(IROpCode::Jmp { delta: 2 });
                    self.add_instruction(IROpCode::LoadBool {
                        dst: $dst,
                        val: false,
                    });
                    self.add_instruction(IROpCode::Jmp { delta: 1 });
                    self.add_instruction(IROpCode::LoadBool {
                        dst: $dst,
                        val: true,
                    });
                }
            };
        }

        match node {
            ExpressionNode::BinaryOperation(lhs, op, rhs) => {
                macro_rules! impl_eq_neq_block {
                    ($eq:expr, $dst:expr, $lhs_type:expr, $rhs_type:expr) => {
                        match ($lhs_type, $rhs_type) {
                            (BytecodeResult::Const(val1), val) => match val {
                                BytecodeResult::Reg(reg) => {
                                    if let Literal::Int(v) = val1 {
                                        if let Ok(v1) = TryInto::<i8>::try_into(v) {
                                            self.add_instruction(IROpCode::EqInt {
                                                src: reg,
                                                int: v1,
                                                res: true,
                                            });

                                            load_bool_result!($dst);
                                            return BytecodeResult::Reg($dst);
                                        }
                                    }
                                    let v1 = self.get_next_vreg();
                                    self.add_result(BytecodeResult::Const(val1), v1);
                                    self.add_instruction(IROpCode::Eq {
                                        lhs: v1,
                                        rhs: reg,
                                        res: $eq,
                                    });
                                    load_bool_result!($dst);
                                    BytecodeResult::Reg($dst)
                                }
                                BytecodeResult::Const(v2) => {
                                    BytecodeResult::Const(Literal::Bool((val1 == v2) == $eq))
                                }
                                _ => unreachable!(),
                            },
                            (BytecodeResult::Reg(v1), BytecodeResult::Reg(v2)) => {
                                self.add_instruction(IROpCode::Eq {
                                    lhs: v1,
                                    rhs: v2,
                                    res: $eq,
                                });
                                load_bool_result!($dst);
                                BytecodeResult::Reg($dst)
                            }
                            _ => unreachable!(),
                        }
                    };
                }

                match op {
                    // Add operation (we can use swap because add is commutative)
                    ast::BinaryOperator::Add => {
                        let lhs_type = self.eval_node(*lhs, None, condition);
                        let mut lhs_type =
                            if matches!(lhs_type, BytecodeResult::Or(_) | BytecodeResult::And(_)) {
                                self.eval_end_or(lhs_type, None)
                            } else {
                                lhs_type
                            };
                        let rhs_type = self.eval_node(*rhs, None, condition);
                        let mut rhs_type =
                            if matches!(rhs_type, BytecodeResult::Or(_) | BytecodeResult::And(_)) {
                                self.eval_end_or(rhs_type, None)
                            } else {
                                rhs_type
                            };

                        if !matches!(lhs_type, BytecodeResult::Reg(_)) {
                            std::mem::swap(&mut lhs_type, &mut rhs_type);
                        }
                        match (lhs_type, rhs_type) {
                            (BytecodeResult::Reg(reg), val) => match val {
                                BytecodeResult::Const(l) => match l {
                                    Literal::Int(k) => {
                                        if let Ok(k) = TryInto::<i8>::try_into(k) {
                                            let res =
                                                dst_reg.unwrap_or_else(|| self.get_next_vreg());
                                            self.add_instruction(IROpCode::AddInt {
                                                src: reg,
                                                dst: res,
                                                int: k,
                                            });
                                            BytecodeResult::Reg(res)
                                        } else {
                                            let c_idx = self.add_constant(l);

                                            if let Ok(c_idx) = TryInto::<u8>::try_into(c_idx) {
                                                let res =
                                                    dst_reg.unwrap_or_else(|| self.get_next_vreg());
                                                self.add_instruction(IROpCode::AddConst {
                                                    src: reg,
                                                    dst: res,
                                                    c_idx,
                                                });
                                                BytecodeResult::Reg(res)
                                            } else {
                                                let v2 = self.get_next_vreg();
                                                self.add_instruction(IROpCode::LoadConst {
                                                    dst: v2,
                                                    c_idx,
                                                });
                                                let res =
                                                    dst_reg.unwrap_or_else(|| self.get_next_vreg());
                                                self.add_instruction(IROpCode::Add {
                                                    lhs: reg,
                                                    rhs: v2,
                                                    dst: res,
                                                });
                                                BytecodeResult::Reg(res)
                                            }
                                        }
                                    }
                                    l => {
                                        let c_idx = self.add_constant(l);

                                        if let Ok(c_idx) = TryInto::<u8>::try_into(c_idx) {
                                            let res =
                                                dst_reg.unwrap_or_else(|| self.get_next_vreg());
                                            self.add_instruction(IROpCode::AddConst {
                                                src: reg,
                                                dst: res,
                                                c_idx,
                                            });
                                            BytecodeResult::Reg(res)
                                        } else {
                                            let v2 = self.get_next_vreg();
                                            self.add_instruction(IROpCode::LoadConst {
                                                dst: v2,
                                                c_idx,
                                            });
                                            let res =
                                                dst_reg.unwrap_or_else(|| self.get_next_vreg());
                                            self.add_instruction(IROpCode::Add {
                                                lhs: reg,
                                                rhs: v2,
                                                dst: res,
                                            });
                                            BytecodeResult::Reg(res)
                                        }
                                    }
                                },
                                BytecodeResult::Reg(rhs) => {
                                    let res = dst_reg.unwrap_or_else(|| self.get_next_vreg());

                                    self.add_instruction(IROpCode::Add {
                                        lhs: reg,
                                        rhs,
                                        dst: res,
                                    });
                                    BytecodeResult::Reg(res)
                                }
                                _ => unreachable!(),
                            },
                            (BytecodeResult::Const(v1), BytecodeResult::Const(v2)) => {
                                match (v1, v2) {
                                    (Literal::Int(v1), Literal::Int(v2)) => {
                                        BytecodeResult::Const(Literal::Int(v1 + v2))
                                    }
                                    (Literal::Int(v1), Literal::Float(v2)) => {
                                        BytecodeResult::Const(Literal::Float(
                                            OrderedFloat(v1 as f64) + v2,
                                        ))
                                    }
                                    (Literal::Float(v1), Literal::Int(v2)) => {
                                        BytecodeResult::Const(Literal::Float(
                                            v1 + OrderedFloat(v2 as f64),
                                        ))
                                    }
                                    (Literal::Float(v1), Literal::Float(v2)) => {
                                        BytecodeResult::Const(Literal::Float(v1 + v2))
                                    }
                                    (Literal::Str(s1), Literal::Str(s2)) => {
                                        BytecodeResult::Const(Literal::Str(Box::new(*s1 + &*s2)))
                                    }
                                    _ => unreachable!(),
                                }
                            }
                            (a, b) => unreachable!("{a:?}, {b:?}"),
                        }
                    }

                    //Multiply operation (we can use swap because mul is commutative)
                    ast::BinaryOperator::Multiply => {
                        let lhs_type = self.eval_node(*lhs, None, condition);
                        let mut lhs_type =
                            if matches!(lhs_type, BytecodeResult::Or(_) | BytecodeResult::And(_)) {
                                self.eval_end_or(lhs_type, None)
                            } else {
                                lhs_type
                            };
                        let rhs_type = self.eval_node(*rhs, None, condition);
                        let mut rhs_type =
                            if matches!(rhs_type, BytecodeResult::Or(_) | BytecodeResult::And(_)) {
                                self.eval_end_or(rhs_type, None)
                            } else {
                                rhs_type
                            };

                        if !matches!(lhs_type, BytecodeResult::Reg(_)) {
                            std::mem::swap(&mut lhs_type, &mut rhs_type);
                        }
                        match (lhs_type, rhs_type) {
                            (BytecodeResult::Reg(reg), val) => match val {
                                BytecodeResult::Const(l) => {
                                    let c_idx = self.add_constant(l);

                                    if let Ok(c_idx) = TryInto::<u8>::try_into(c_idx) {
                                        let res = dst_reg.unwrap_or_else(|| self.get_next_vreg());
                                        self.add_instruction(IROpCode::MulConst {
                                            src: reg,
                                            dst: res,
                                            c_idx,
                                        });
                                        BytecodeResult::Reg(res)
                                    } else {
                                        let v2 = self.get_next_vreg();
                                        self.add_instruction(IROpCode::LoadConst {
                                            dst: v2,
                                            c_idx,
                                        });
                                        let res = dst_reg.unwrap_or_else(|| self.get_next_vreg());
                                        self.add_instruction(IROpCode::Mul {
                                            lhs: reg,
                                            rhs: v2,
                                            dst: res,
                                        });
                                        BytecodeResult::Reg(res)
                                    }
                                }
                                BytecodeResult::Reg(rhs) => {
                                    let res = dst_reg.unwrap_or_else(|| self.get_next_vreg());

                                    self.add_instruction(IROpCode::Mul {
                                        lhs: reg,
                                        rhs,
                                        dst: res,
                                    });
                                    BytecodeResult::Reg(res)
                                }
                                _ => unreachable!(),
                            },
                            (a, b) => unreachable!("{a:?}, {b:?}"),
                        }
                    }

                    // Substract operation
                    ast::BinaryOperator::Substract => {
                        let lhs_type = self.eval_node(*lhs, None, condition);
                        let lhs_type =
                            if matches!(lhs_type, BytecodeResult::Or(_) | BytecodeResult::And(_)) {
                                self.eval_end_or(lhs_type, None)
                            } else {
                                lhs_type
                            };
                        let rhs_type = self.eval_node(*rhs, None, condition);
                        let rhs_type =
                            if matches!(rhs_type, BytecodeResult::Or(_) | BytecodeResult::And(_)) {
                                self.eval_end_or(rhs_type, None)
                            } else {
                                rhs_type
                            };

                        match (lhs_type, rhs_type) {
                            (BytecodeResult::Reg(reg), val) => match val {
                                BytecodeResult::Const(l) => {
                                    let l = match l {
                                        Literal::Int(v) => Literal::Int(-v),
                                        Literal::Float(v) => Literal::Float(-v),
                                        _ => unreachable!(),
                                    };
                                    match l {
                                        Literal::Int(k) => {
                                            if let Ok(k) = TryInto::<i8>::try_into(k) {
                                                let res =
                                                    dst_reg.unwrap_or_else(|| self.get_next_vreg());
                                                self.add_instruction(IROpCode::AddInt {
                                                    src: reg,
                                                    dst: res,
                                                    int: k,
                                                });
                                                BytecodeResult::Reg(res)
                                            } else {
                                                let c_idx = self.add_constant(l);

                                                if let Ok(c_idx) = TryInto::<u8>::try_into(c_idx) {
                                                    let res = dst_reg
                                                        .unwrap_or_else(|| self.get_next_vreg());
                                                    self.add_instruction(IROpCode::AddConst {
                                                        src: reg,
                                                        dst: res,
                                                        c_idx,
                                                    });
                                                    BytecodeResult::Reg(res)
                                                } else {
                                                    let v2 = self.get_next_vreg();
                                                    self.add_instruction(IROpCode::LoadConst {
                                                        dst: v2,
                                                        c_idx,
                                                    });
                                                    let res = dst_reg
                                                        .unwrap_or_else(|| self.get_next_vreg());
                                                    self.add_instruction(IROpCode::Add {
                                                        lhs: reg,
                                                        rhs: v2,
                                                        dst: res,
                                                    });
                                                    BytecodeResult::Reg(res)
                                                }
                                            }
                                        }
                                        l => {
                                            let c_idx = self.add_constant(l);

                                            if let Ok(c_idx) = TryInto::<u8>::try_into(c_idx) {
                                                let res =
                                                    dst_reg.unwrap_or_else(|| self.get_next_vreg());
                                                self.add_instruction(IROpCode::AddConst {
                                                    src: reg,
                                                    dst: res,
                                                    c_idx,
                                                });
                                                BytecodeResult::Reg(res)
                                            } else {
                                                let v2 = self.get_next_vreg();
                                                self.add_instruction(IROpCode::LoadConst {
                                                    dst: v2,
                                                    c_idx,
                                                });
                                                let res =
                                                    dst_reg.unwrap_or_else(|| self.get_next_vreg());
                                                self.add_instruction(IROpCode::Add {
                                                    lhs: reg,
                                                    rhs: v2,
                                                    dst: res,
                                                });
                                                BytecodeResult::Reg(res)
                                            }
                                        }
                                    }
                                }
                                BytecodeResult::Reg(rhs) => {
                                    let res = dst_reg.unwrap_or_else(|| self.get_next_vreg());

                                    self.add_instruction(IROpCode::Sub {
                                        lhs: reg,
                                        rhs,
                                        dst: res,
                                    });
                                    BytecodeResult::Reg(res)
                                }
                                _ => unreachable!(),
                            },
                            (BytecodeResult::Const(v1), val) => match val {
                                BytecodeResult::Const(v2) => match (v1, v2) {
                                    (Literal::Int(v1), Literal::Int(v2)) => {
                                        BytecodeResult::Const(Literal::Int(v1 - v2))
                                    }
                                    (Literal::Int(v1), Literal::Float(v2)) => {
                                        BytecodeResult::Const(Literal::Float(
                                            OrderedFloat(v1 as f64) - v2,
                                        ))
                                    }
                                    (Literal::Float(v1), Literal::Int(v2)) => {
                                        BytecodeResult::Const(Literal::Float(
                                            v1 - OrderedFloat(v2 as f64),
                                        ))
                                    }
                                    (Literal::Float(v1), Literal::Float(v2)) => {
                                        BytecodeResult::Const(Literal::Float(v1 - v2))
                                    }
                                    _ => unreachable!(),
                                },
                                BytecodeResult::Reg(reg) => {
                                    let c_idx = self.add_constant(v1);

                                    let v1 = self.get_next_vreg();
                                    self.add_instruction(IROpCode::LoadConst { dst: v1, c_idx });
                                    let res = dst_reg.unwrap_or_else(|| self.get_next_vreg());
                                    self.add_instruction(IROpCode::Sub {
                                        lhs: v1,
                                        rhs: reg,
                                        dst: res,
                                    });
                                    BytecodeResult::Reg(res)
                                }
                                _ => unreachable!(),
                            },
                            _ => unreachable!(),
                        }
                    }

                    // Divide operation
                    ast::BinaryOperator::Divide => {
                        let lhs_type = self.eval_node(*lhs, None, condition);
                        let lhs_type =
                            if matches!(lhs_type, BytecodeResult::Or(_) | BytecodeResult::And(_)) {
                                self.eval_end_or(lhs_type, None)
                            } else {
                                lhs_type
                            };
                        let rhs_type = self.eval_node(*rhs, None, condition);
                        let rhs_type =
                            if matches!(rhs_type, BytecodeResult::Or(_) | BytecodeResult::And(_)) {
                                self.eval_end_or(rhs_type, None)
                            } else {
                                rhs_type
                            };

                        match (lhs_type, rhs_type) {
                            (BytecodeResult::Reg(reg), val) => match val {
                                BytecodeResult::Const(l) => {
                                    let c_idx = self.add_constant(l);

                                    if let Ok(c_idx) = TryInto::<u8>::try_into(c_idx) {
                                        let res = dst_reg.unwrap_or_else(|| self.get_next_vreg());
                                        self.add_instruction(IROpCode::DivConst {
                                            src: reg,
                                            dst: res,
                                            c_idx,
                                        });
                                        BytecodeResult::Reg(res)
                                    } else {
                                        let v2 = self.get_next_vreg();
                                        self.add_instruction(IROpCode::LoadConst {
                                            dst: v2,
                                            c_idx,
                                        });
                                        let res = dst_reg.unwrap_or_else(|| self.get_next_vreg());
                                        self.add_instruction(IROpCode::Div {
                                            lhs: reg,
                                            rhs: v2,
                                            dst: res,
                                        });
                                        BytecodeResult::Reg(res)
                                    }
                                }
                                BytecodeResult::Reg(rhs) => {
                                    let res = dst_reg.unwrap_or_else(|| self.get_next_vreg());

                                    self.add_instruction(IROpCode::Div {
                                        lhs: reg,
                                        rhs,
                                        dst: res,
                                    });
                                    BytecodeResult::Reg(res)
                                }
                                _ => unreachable!(),
                            },
                            (BytecodeResult::Const(v1), val) => match val {
                                BytecodeResult::Const(v2) => match (v1, v2) {
                                    (Literal::Int(v1), Literal::Int(v2)) => {
                                        BytecodeResult::Const(Literal::Int(v1 / v2))
                                    }
                                    (Literal::Int(v1), Literal::Float(v2)) => {
                                        BytecodeResult::Const(Literal::Float(
                                            OrderedFloat(v1 as f64) / v2,
                                        ))
                                    }
                                    (Literal::Float(v1), Literal::Int(v2)) => {
                                        BytecodeResult::Const(Literal::Float(
                                            v1 / OrderedFloat(v2 as f64),
                                        ))
                                    }
                                    (Literal::Float(v1), Literal::Float(v2)) => {
                                        BytecodeResult::Const(Literal::Float(v1 / v2))
                                    }
                                    _ => unreachable!(),
                                },
                                BytecodeResult::Reg(reg) => {
                                    let c_idx = self.add_constant(v1);

                                    let v1 = self.get_next_vreg();
                                    self.add_instruction(IROpCode::LoadConst { dst: v1, c_idx });
                                    let res = dst_reg.unwrap_or_else(|| self.get_next_vreg());
                                    self.add_instruction(IROpCode::Div {
                                        lhs: v1,
                                        rhs: reg,
                                        dst: res,
                                    });
                                    BytecodeResult::Reg(res)
                                }
                                _ => unreachable!(),
                            },
                            _ => unreachable!(),
                        }
                    }
                    ast::BinaryOperator::Equal => {
                        let lhs_type = self.eval_node(*lhs, None, condition);
                        let mut lhs_type =
                            if matches!(lhs_type, BytecodeResult::Or(_) | BytecodeResult::And(_)) {
                                self.eval_end_or(lhs_type, None)
                            } else {
                                lhs_type
                            };
                        let rhs_type = self.eval_node(*rhs, None, condition);
                        let mut rhs_type =
                            if matches!(rhs_type, BytecodeResult::Or(_) | BytecodeResult::And(_)) {
                                self.eval_end_or(rhs_type, None)
                            } else {
                                rhs_type
                            };

                        if !matches!(lhs_type, BytecodeResult::Const(_)) {
                            std::mem::swap(&mut lhs_type, &mut rhs_type);
                        }
                        let res = dst_reg.unwrap_or_else(|| self.get_next_vreg());
                        impl_eq_neq_block!(true, res, lhs_type, rhs_type)
                    }
                    ast::BinaryOperator::NotEqual => {
                        let lhs_type = self.eval_node(*lhs, None, condition);
                        let mut lhs_type = self.eval_end_or(lhs_type, None);
                        let rhs_type = self.eval_node(*rhs, None, condition);
                        let mut rhs_type = self.eval_end_or(rhs_type, None);

                        if !matches!(lhs_type, BytecodeResult::Const(_)) {
                            std::mem::swap(&mut lhs_type, &mut rhs_type);
                        }
                        let res = dst_reg.unwrap_or_else(|| self.get_next_vreg());
                        impl_eq_neq_block!(false, res, lhs_type, rhs_type)
                    }
                    ast::BinaryOperator::GreaterThan => {
                        let lhs_type = self.eval_node(*lhs, None, condition);
                        let lhs_type =
                            if matches!(lhs_type, BytecodeResult::Or(_) | BytecodeResult::And(_)) {
                                self.eval_end_or(lhs_type, None)
                            } else {
                                lhs_type
                            };
                        let rhs_type = self.eval_node(*rhs, None, condition);
                        let rhs_type =
                            if matches!(rhs_type, BytecodeResult::Or(_) | BytecodeResult::And(_)) {
                                self.eval_end_or(rhs_type, None)
                            } else {
                                rhs_type
                            };

                        let res = self.get_next_vreg();

                        let result = match (lhs_type, rhs_type) {
                            (BytecodeResult::Const(v1), val) => match val {
                                BytecodeResult::Reg(reg) => {
                                    if let Literal::Int(v) = v1 {
                                        if let Ok(v1) = TryInto::<i8>::try_into(v) {
                                            self.add_instruction(IROpCode::LtInt {
                                                src: reg,
                                                int: v1,
                                                res: true,
                                            });

                                            load_bool_result!(res);
                                            return BytecodeResult::Reg(res);
                                        }
                                    }
                                    let c_idx = self.add_constant(v1);

                                    if let Ok(c_idx) = TryInto::<u8>::try_into(c_idx) {
                                        self.add_instruction(IROpCode::LtConst {
                                            src: reg,
                                            c_idx,
                                            res: true,
                                        });
                                        BytecodeResult::Reg(res)
                                    } else {
                                        let v1 = self.get_next_vreg();
                                        self.add_instruction(IROpCode::LoadConst {
                                            dst: v1,
                                            c_idx,
                                        });

                                        self.add_instruction(IROpCode::Lt {
                                            lhs: reg,
                                            rhs: v1,
                                            res: true,
                                        });
                                        BytecodeResult::Reg(res)
                                    }
                                }
                                BytecodeResult::Const(v2) => {
                                    BytecodeResult::Const(Literal::Bool(match (v1, v2) {
                                        (Literal::Int(v1), Literal::Int(v2)) => v1 > v2,
                                        (Literal::Float(v1), Literal::Int(v2)) => {
                                            v1 > OrderedFloat(v2 as f64)
                                        }
                                        (Literal::Int(v1), Literal::Float(v2)) => {
                                            OrderedFloat(v1 as f64) > v2
                                        }
                                        (Literal::Float(v1), Literal::Float(v2)) => v1 > v2,
                                        _ => unreachable!(),
                                    }))
                                }
                                _ => unreachable!(),
                            },
                            (BytecodeResult::Reg(v1), val) => match val {
                                BytecodeResult::Reg(v2) => {
                                    self.add_instruction(IROpCode::Lt {
                                        lhs: v2,
                                        rhs: v1,
                                        res: true,
                                    });

                                    BytecodeResult::Reg(res)
                                }
                                BytecodeResult::Const(v2) => {
                                    if let Literal::Int(v) = v2 {
                                        if let Ok(v2) = TryInto::<i8>::try_into(v) {
                                            self.add_instruction(IROpCode::GtInt {
                                                src: v1,
                                                int: v2,
                                                res: true,
                                            });

                                            load_bool_result!(res);
                                            return BytecodeResult::Reg(res);
                                        }
                                    }
                                    let c_idx = self.add_constant(v2);

                                    if let Ok(c_idx) = TryInto::<u8>::try_into(c_idx) {
                                        self.add_instruction(IROpCode::GtConst {
                                            src: v1,
                                            c_idx,
                                            res: true,
                                        });
                                        BytecodeResult::Reg(res)
                                    } else {
                                        let v2 = self.get_next_vreg();
                                        self.add_instruction(IROpCode::LoadConst {
                                            dst: v2,
                                            c_idx,
                                        });

                                        self.add_instruction(IROpCode::Lt {
                                            lhs: v2,
                                            rhs: v1,
                                            res: true,
                                        });
                                        BytecodeResult::Reg(res)
                                    }
                                }
                                _ => unreachable!(),
                            },
                            _ => unreachable!(),
                        };
                        if matches!(result, BytecodeResult::Reg(_)) {
                            load_bool_result!(res);
                        }
                        result
                    }

                    ast::BinaryOperator::GreaterThanOrEqual => {
                        let lhs_type = self.eval_node(*lhs, None, condition);
                        let lhs_type =
                            if matches!(lhs_type, BytecodeResult::Or(_) | BytecodeResult::And(_)) {
                                self.eval_end_or(lhs_type, None)
                            } else {
                                lhs_type
                            };
                        let rhs_type = self.eval_node(*rhs, None, condition);
                        let rhs_type =
                            if matches!(rhs_type, BytecodeResult::Or(_) | BytecodeResult::And(_)) {
                                self.eval_end_or(rhs_type, None)
                            } else {
                                rhs_type
                            };

                        let res = self.get_next_vreg();

                        let result = match (lhs_type, rhs_type) {
                            (BytecodeResult::Const(v1), val) => match val {
                                BytecodeResult::Reg(reg) => {
                                    if let Literal::Int(v) = v1 {
                                        if let Ok(v1) = TryInto::<i8>::try_into(v) {
                                            self.add_instruction(IROpCode::LteInt {
                                                src: reg,
                                                int: v1,
                                                res: true,
                                            });

                                            load_bool_result!(res);
                                            return BytecodeResult::Reg(res);
                                        }
                                    }
                                    let c_idx = self.add_constant(v1);

                                    if let Ok(c_idx) = TryInto::<u8>::try_into(c_idx) {
                                        self.add_instruction(IROpCode::LteConst {
                                            src: reg,
                                            c_idx,
                                            res: true,
                                        });
                                        BytecodeResult::Reg(res)
                                    } else {
                                        let v1 = self.get_next_vreg();
                                        self.add_instruction(IROpCode::LoadConst {
                                            dst: v1,
                                            c_idx,
                                        });

                                        self.add_instruction(IROpCode::Lte {
                                            lhs: reg,
                                            rhs: v1,
                                            res: true,
                                        });
                                        BytecodeResult::Reg(res)
                                    }
                                }
                                BytecodeResult::Const(v2) => {
                                    BytecodeResult::Const(Literal::Bool(match (v1, v2) {
                                        (Literal::Int(v1), Literal::Int(v2)) => v1 >= v2,
                                        (Literal::Float(v1), Literal::Int(v2)) => {
                                            v1 >= OrderedFloat(v2 as f64)
                                        }
                                        (Literal::Int(v1), Literal::Float(v2)) => {
                                            OrderedFloat(v1 as f64) >= v2
                                        }
                                        (Literal::Float(v1), Literal::Float(v2)) => v1 >= v2,
                                        _ => unreachable!(),
                                    }))
                                }
                                _ => unreachable!(),
                            },
                            (BytecodeResult::Reg(v1), val) => match val {
                                BytecodeResult::Reg(v2) => {
                                    self.add_instruction(IROpCode::Lte {
                                        lhs: v2,
                                        rhs: v1,
                                        res: true,
                                    });

                                    BytecodeResult::Reg(res)
                                }
                                BytecodeResult::Const(v2) => {
                                    if let Literal::Int(v) = v2 {
                                        if let Ok(v2) = TryInto::<i8>::try_into(v) {
                                            self.add_instruction(IROpCode::GteInt {
                                                src: v1,
                                                int: v2,
                                                res: true,
                                            });

                                            load_bool_result!(res);
                                            return BytecodeResult::Reg(res);
                                        }
                                    }
                                    let c_idx = self.add_constant(v2);

                                    if let Ok(c_idx) = TryInto::<u8>::try_into(c_idx) {
                                        self.add_instruction(IROpCode::GteConst {
                                            src: v1,
                                            c_idx,
                                            res: true,
                                        });
                                        BytecodeResult::Reg(res)
                                    } else {
                                        let v2 = self.get_next_vreg();
                                        self.add_instruction(IROpCode::LoadConst {
                                            dst: v2,
                                            c_idx,
                                        });

                                        self.add_instruction(IROpCode::Lte {
                                            lhs: v2,
                                            rhs: v1,
                                            res: true,
                                        });
                                        BytecodeResult::Reg(res)
                                    }
                                }
                                _ => unreachable!(),
                            },
                            _ => unreachable!(),
                        };
                        if matches!(result, BytecodeResult::Reg(_)) {
                            load_bool_result!(res);
                        }
                        result
                    }

                    ast::BinaryOperator::LessThan => {
                        let lhs_type = self.eval_node(*lhs, None, condition);
                        let lhs_type =
                            if matches!(lhs_type, BytecodeResult::Or(_) | BytecodeResult::And(_)) {
                                self.eval_end_or(lhs_type, None)
                            } else {
                                lhs_type
                            };
                        let rhs_type = self.eval_node(*rhs, None, condition);
                        let rhs_type =
                            if matches!(rhs_type, BytecodeResult::Or(_) | BytecodeResult::And(_)) {
                                self.eval_end_or(rhs_type, None)
                            } else {
                                rhs_type
                            };

                        let res = self.get_next_vreg();

                        let result = match (lhs_type, rhs_type) {
                            (BytecodeResult::Const(v1), val) => match val {
                                BytecodeResult::Reg(reg) => {
                                    if let Literal::Int(v) = v1 {
                                        if let Ok(v1) = TryInto::<i8>::try_into(v) {
                                            self.add_instruction(IROpCode::GtInt {
                                                src: reg,
                                                int: v1,
                                                res: true,
                                            });

                                            load_bool_result!(res);
                                            return BytecodeResult::Reg(res);
                                        }
                                    }
                                    let c_idx = self.add_constant(v1);

                                    if let Ok(c_idx) = TryInto::<u8>::try_into(c_idx) {
                                        self.add_instruction(IROpCode::GtConst {
                                            src: reg,
                                            c_idx,
                                            res: true,
                                        });
                                        BytecodeResult::Reg(res)
                                    } else {
                                        let v1 = self.get_next_vreg();
                                        self.add_instruction(IROpCode::LoadConst {
                                            dst: v1,
                                            c_idx,
                                        });

                                        self.add_instruction(IROpCode::Lt {
                                            lhs: v1,
                                            rhs: reg,
                                            res: true,
                                        });
                                        BytecodeResult::Reg(res)
                                    }
                                }
                                BytecodeResult::Const(v2) => {
                                    BytecodeResult::Const(Literal::Bool(match (v1, v2) {
                                        (Literal::Int(v1), Literal::Int(v2)) => v1 < v2,
                                        (Literal::Float(v1), Literal::Int(v2)) => {
                                            v1 < OrderedFloat(v2 as f64)
                                        }
                                        (Literal::Int(v1), Literal::Float(v2)) => {
                                            OrderedFloat(v1 as f64) < v2
                                        }
                                        (Literal::Float(v1), Literal::Float(v2)) => v1 < v2,
                                        _ => unreachable!(),
                                    }))
                                }
                                _ => unreachable!(),
                            },
                            (BytecodeResult::Reg(v1), val) => match val {
                                BytecodeResult::Reg(v2) => {
                                    self.add_instruction(IROpCode::Lt {
                                        lhs: v1,
                                        rhs: v2,
                                        res: true,
                                    });

                                    BytecodeResult::Reg(res)
                                }
                                BytecodeResult::Const(v2) => {
                                    if let Literal::Int(v) = v2 {
                                        if let Ok(v2) = TryInto::<i8>::try_into(v) {
                                            self.add_instruction(IROpCode::LtInt {
                                                src: v1,
                                                int: v2,
                                                res: true,
                                            });

                                            load_bool_result!(res);
                                            return BytecodeResult::Reg(res);
                                        }
                                    }
                                    let c_idx = self.add_constant(v2);

                                    if let Ok(c_idx) = TryInto::<u8>::try_into(c_idx) {
                                        self.add_instruction(IROpCode::LtConst {
                                            src: v1,
                                            c_idx,
                                            res: true,
                                        });
                                        BytecodeResult::Reg(res)
                                    } else {
                                        let v2 = self.get_next_vreg();
                                        self.add_instruction(IROpCode::LoadConst {
                                            dst: v2,
                                            c_idx,
                                        });

                                        self.add_instruction(IROpCode::Lt {
                                            lhs: v1,
                                            rhs: v2,
                                            res: true,
                                        });
                                        BytecodeResult::Reg(res)
                                    }
                                }
                                _ => unreachable!(),
                            },
                            _ => unreachable!(),
                        };
                        if matches!(result, BytecodeResult::Reg(_)) {
                            load_bool_result!(res);
                        }
                        result
                    }

                    ast::BinaryOperator::LessThanOrEqual => {
                        let lhs_type = self.eval_node(*lhs, None, condition);
                        let lhs_type =
                            if matches!(lhs_type, BytecodeResult::Or(_) | BytecodeResult::And(_)) {
                                self.eval_end_or(lhs_type, None)
                            } else {
                                lhs_type
                            };
                        let rhs_type = self.eval_node(*rhs, None, condition);
                        let rhs_type =
                            if matches!(rhs_type, BytecodeResult::Or(_) | BytecodeResult::And(_)) {
                                self.eval_end_or(rhs_type, None)
                            } else {
                                rhs_type
                            };

                        let res = self.get_next_vreg();

                        let result = match (lhs_type, rhs_type) {
                            (BytecodeResult::Const(v1), val) => match val {
                                BytecodeResult::Reg(reg) => {
                                    if let Literal::Int(v) = v1 {
                                        if let Ok(v1) = TryInto::<i8>::try_into(v) {
                                            self.add_instruction(IROpCode::GteInt {
                                                src: reg,
                                                int: v1,
                                                res: true,
                                            });

                                            load_bool_result!(res);
                                            return BytecodeResult::Reg(res);
                                        }
                                    }
                                    let c_idx = self.add_constant(v1);

                                    if let Ok(c_idx) = TryInto::<u8>::try_into(c_idx) {
                                        self.add_instruction(IROpCode::GteConst {
                                            src: reg,
                                            c_idx,
                                            res: true,
                                        });
                                        BytecodeResult::Reg(res)
                                    } else {
                                        let v1 = self.get_next_vreg();
                                        self.add_instruction(IROpCode::LoadConst {
                                            dst: v1,
                                            c_idx,
                                        });

                                        self.add_instruction(IROpCode::Lte {
                                            lhs: v1,
                                            rhs: reg,
                                            res: true,
                                        });
                                        BytecodeResult::Reg(res)
                                    }
                                }
                                BytecodeResult::Const(v2) => {
                                    BytecodeResult::Const(Literal::Bool(match (v1, v2) {
                                        (Literal::Int(v1), Literal::Int(v2)) => v1 <= v2,
                                        (Literal::Float(v1), Literal::Int(v2)) => {
                                            v1 <= OrderedFloat(v2 as f64)
                                        }
                                        (Literal::Int(v1), Literal::Float(v2)) => {
                                            OrderedFloat(v1 as f64) <= v2
                                        }
                                        (Literal::Float(v1), Literal::Float(v2)) => v1 <= v2,
                                        _ => unreachable!(),
                                    }))
                                }
                                _ => unreachable!(),
                            },
                            (BytecodeResult::Reg(v1), val) => match val {
                                BytecodeResult::Reg(v2) => {
                                    self.add_instruction(IROpCode::Lte {
                                        lhs: v1,
                                        rhs: v2,
                                        res: true,
                                    });

                                    BytecodeResult::Reg(res)
                                }
                                BytecodeResult::Const(v2) => {
                                    if let Literal::Int(v) = v2 {
                                        if let Ok(v2) = TryInto::<i8>::try_into(v) {
                                            self.add_instruction(IROpCode::LteInt {
                                                src: v1,
                                                int: v2,
                                                res: true,
                                            });

                                            load_bool_result!(res);
                                            return BytecodeResult::Reg(res);
                                        }
                                    }
                                    let c_idx = self.add_constant(v2);

                                    if let Ok(c_idx) = TryInto::<u8>::try_into(c_idx) {
                                        self.add_instruction(IROpCode::LteConst {
                                            src: v1,
                                            c_idx,
                                            res: true,
                                        });
                                        BytecodeResult::Reg(res)
                                    } else {
                                        let v2 = self.get_next_vreg();
                                        self.add_instruction(IROpCode::LoadConst {
                                            dst: v2,
                                            c_idx,
                                        });

                                        self.add_instruction(IROpCode::Lte {
                                            lhs: v1,
                                            rhs: v2,
                                            res: true,
                                        });
                                        BytecodeResult::Reg(res)
                                    }
                                }
                                _ => unreachable!(),
                            },
                            _ => unreachable!(),
                        };
                        if matches!(result, BytecodeResult::Reg(_)) {
                            load_bool_result!(res);
                        }
                        result
                    }
                    ast::BinaryOperator::LogicalOr => {
                        let mut lhs_type = self.eval_node(*lhs, None, condition);
                        let mut rhs_type = self.eval_node(*rhs, None, condition);
                        if !matches!(lhs_type, BytecodeResult::Or(_) | BytecodeResult::Reg(_)) {
                            std::mem::swap(&mut lhs_type, &mut rhs_type);
                        }
                        match (lhs_type, rhs_type) {
                            (BytecodeResult::Or(mut v1), BytecodeResult::Or(v2)) => {
                                BytecodeResult::Or({
                                    v1.extend(v2);
                                    v1
                                })
                            }
                            (BytecodeResult::Or(mut v1), v2) => match v2 {
                                BytecodeResult::Const(Literal::Bool(v2)) => {
                                    if v2 {
                                        BytecodeResult::Const(Literal::Bool(true))
                                    } else {
                                        BytecodeResult::Or(v1)
                                    }
                                }
                                v2 => BytecodeResult::Or({
                                    v1.push(v2);
                                    v1
                                }),
                            },
                            (BytecodeResult::Reg(v1), BytecodeResult::Const(Literal::Bool(v2))) => {
                                if v2 {
                                    BytecodeResult::Const(Literal::Bool(true))
                                } else {
                                    BytecodeResult::Reg(v1)
                                }
                            }
                            (
                                BytecodeResult::Const(Literal::Bool(v1)),
                                BytecodeResult::Const(Literal::Bool(v2)),
                            ) => BytecodeResult::Const(Literal::Bool(v1 || v2)),
                            (v1, v2) => BytecodeResult::Or(vec![v1, v2]),
                        }
                    }
                    ast::BinaryOperator::LogicalAnd => {
                        let mut lhs_type = self.eval_node(*lhs, None, condition);
                        let mut rhs_type = self.eval_node(*rhs, None, condition);
                        if !matches!(lhs_type, BytecodeResult::And(_) | BytecodeResult::Reg(_)) {
                            std::mem::swap(&mut lhs_type, &mut rhs_type);
                        }
                        match (lhs_type, rhs_type) {
                            (BytecodeResult::And(mut v1), BytecodeResult::And(v2)) => {
                                BytecodeResult::And({
                                    v1.extend(v2);
                                    v1
                                })
                            }
                            (BytecodeResult::And(mut v1), v2) => match v2 {
                                BytecodeResult::Const(Literal::Bool(v2)) => {
                                    if !v2 {
                                        BytecodeResult::Const(Literal::Bool(false))
                                    } else {
                                        BytecodeResult::And(v1)
                                    }
                                }
                                v2 => BytecodeResult::And({
                                    v1.push(v2);
                                    v1
                                }),
                            },
                            (BytecodeResult::Reg(v1), BytecodeResult::Const(Literal::Bool(v2))) => {
                                if !v2 {
                                    BytecodeResult::Const(Literal::Bool(false))
                                } else {
                                    BytecodeResult::Reg(v1)
                                }
                            }
                            (
                                BytecodeResult::Const(Literal::Bool(v1)),
                                BytecodeResult::Const(Literal::Bool(v2)),
                            ) => BytecodeResult::Const(Literal::Bool(v1 && v2)),
                            (v1, v2) => BytecodeResult::And(vec![v1, v2]),
                        }
                    }
                    ast::BinaryOperator::Mod => {
                        let lhs_type = self.eval_node(*lhs, None, condition);
                        let lhs_type =
                            if matches!(lhs_type, BytecodeResult::Or(_) | BytecodeResult::And(_)) {
                                self.eval_end_or(lhs_type, None)
                            } else {
                                lhs_type
                            };
                        let rhs_type = self.eval_node(*rhs, None, condition);
                        let rhs_type =
                            if matches!(rhs_type, BytecodeResult::Or(_) | BytecodeResult::And(_)) {
                                self.eval_end_or(rhs_type, None)
                            } else {
                                rhs_type
                            };

                        match (lhs_type, rhs_type) {
                            (BytecodeResult::Reg(reg), val) => match val {
                                BytecodeResult::Const(l) => {
                                    let c_idx = self.add_constant(l);

                                    if let Ok(c_idx) = TryInto::<u8>::try_into(c_idx) {
                                        let res = dst_reg.unwrap_or_else(|| self.get_next_vreg());
                                        self.add_instruction(IROpCode::ModConst {
                                            lhs: reg,
                                            dst: res,
                                            c_idx,
                                        });
                                        BytecodeResult::Reg(res)
                                    } else {
                                        let v2 = self.get_next_vreg();
                                        self.add_instruction(IROpCode::LoadConst {
                                            dst: v2,
                                            c_idx,
                                        });
                                        let res = dst_reg.unwrap_or_else(|| self.get_next_vreg());
                                        self.add_instruction(IROpCode::Mod {
                                            lhs: reg,
                                            rhs: v2,
                                            dst: res,
                                        });
                                        BytecodeResult::Reg(res)
                                    }
                                }
                                BytecodeResult::Reg(rhs) => {
                                    let res = dst_reg.unwrap_or_else(|| self.get_next_vreg());

                                    self.add_instruction(IROpCode::Mod {
                                        lhs: reg,
                                        rhs,
                                        dst: res,
                                    });
                                    BytecodeResult::Reg(res)
                                }
                                _ => unreachable!(),
                            },
                            (BytecodeResult::Const(v1), val) => match val {
                                BytecodeResult::Const(v2) => match (v1, v2) {
                                    (Literal::Int(v1), Literal::Int(v2)) => {
                                        BytecodeResult::Const(Literal::Int(v1 % v2))
                                    }
                                    (Literal::Int(v1), Literal::Float(v2)) => {
                                        BytecodeResult::Const(Literal::Float(
                                            OrderedFloat(v1 as f64) % v2,
                                        ))
                                    }
                                    (Literal::Float(v1), Literal::Int(v2)) => {
                                        BytecodeResult::Const(Literal::Float(
                                            v1 % OrderedFloat(v2 as f64),
                                        ))
                                    }
                                    (Literal::Float(v1), Literal::Float(v2)) => {
                                        BytecodeResult::Const(Literal::Float(v1 % v2))
                                    }
                                    _ => unreachable!(),
                                },
                                BytecodeResult::Reg(reg) => {
                                    let c_idx = self.add_constant(v1);

                                    let v1 = self.get_next_vreg();
                                    self.add_instruction(IROpCode::LoadConst { dst: v1, c_idx });
                                    let res = dst_reg.unwrap_or_else(|| self.get_next_vreg());
                                    self.add_instruction(IROpCode::Mod {
                                        lhs: v1,
                                        rhs: reg,
                                        dst: res,
                                    });
                                    BytecodeResult::Reg(res)
                                }
                                _ => unreachable!(),
                            },
                            _ => unreachable!(),
                        }
                    }
                    ast::BinaryOperator::Power => todo!(),
                    ast::BinaryOperator::Div => todo!(),
                }
            }
            ExpressionNode::UnaryOperation(op, rhs) => {
                let rhs_type = self.eval_node(*rhs, None, condition);
                let res = dst_reg.unwrap_or_else(|| self.get_next_vreg());

                let rhs_type = if matches!(rhs_type, BytecodeResult::Or(_) | BytecodeResult::And(_))
                {
                    self.eval_end_or(rhs_type, Some(res))
                } else {
                    rhs_type
                };
                match op {
                    ast::UnaryOperator::Negate => match rhs_type {
                        BytecodeResult::Reg(v) => {
                            self.add_instruction(IROpCode::UnaryMinus { src: v, dst: res });
                            BytecodeResult::Reg(res)
                        }
                        BytecodeResult::Const(c) => match c {
                            Literal::Int(v) => BytecodeResult::Const(Literal::Int(-v)),
                            Literal::Float(v) => BytecodeResult::Const(Literal::Float(-v)),
                            _ => unreachable!(),
                        },
                        _ => unreachable!(),
                    },
                    ast::UnaryOperator::LogicalNot => match rhs_type {
                        BytecodeResult::Const(v) => BytecodeResult::Const(match v {
                            Literal::Bool(v) => Literal::Bool(!v),
                            _ => unreachable!(),
                        }),
                        BytecodeResult::Reg(v) => {
                            let res = dst_reg.unwrap_or_else(|| self.get_next_vreg());

                            self.add_instruction(IROpCode::Not { src: v, dst: res });
                            BytecodeResult::Reg(res)
                        }
                        _ => unreachable!(),
                    },
                    ast::UnaryOperator::Index(exp) => {
                        let exp_type = self.eval_node(*exp, None, condition);
                        let exp_type =
                            if matches!(exp_type, BytecodeResult::Or(_) | BytecodeResult::And(_)) {
                                self.eval_end_or(exp_type, Some(res))
                            } else {
                                exp_type
                            };
                        match rhs_type {
                            BytecodeResult::Reg(table) => match exp_type {
                                BytecodeResult::Reg(v_idx) => {
                                    self.add_instruction(IROpCode::GetIndex {
                                        dst: res,
                                        t_src: table,
                                        v_src: v_idx,
                                    });
                                }
                                BytecodeResult::Const(k) => match k {
                                    Literal::Int(k) => {
                                        if let Ok(val) = TryInto::<i8>::try_into(k) {
                                            self.add_instruction(IROpCode::GetIntIndex {
                                                dst: res,
                                                t_src: table,
                                                int: val,
                                            });
                                        } else {
                                            let v_idx: u8 =
                                                self.add_constant(Literal::Int(k)) as u8;
                                            self.add_instruction(IROpCode::GetConstIndex {
                                                dst: res,
                                                t_src: table,
                                                v_idx,
                                            });
                                        }
                                    }
                                    k => {
                                        let v_idx: u8 = self.add_constant(k) as u8;
                                        self.add_instruction(IROpCode::GetConstIndex {
                                            dst: res,
                                            t_src: table,
                                            v_idx,
                                        });
                                    }
                                },
                                _ => unreachable!(),
                            },
                            _ => unreachable!(),
                        }
                        BytecodeResult::Reg(res)
                    }
                }
            }
            ExpressionNode::FunctionCall(FunctionCallNode {
                identifier,
                arguments,
            }) => {
                let mut res = dst_reg.unwrap_or_else(|| self.get_next_vreg());
                if let Some(dst_reg) = dst_reg {
                    if dst_reg != self.next_reg - 1 {
                        res = self.get_next_vreg();
                    }
                }
                let rf = self.eval_node(*identifier, Some(res), false);
                let result = match rf {
                    BytecodeResult::Reg(reg) => {
                        let args_len = arguments.len();
                        self.fixed_regs.insert(
                            self.instructions.len() - 1,
                            reg.min(self.max_reg - args_len - self.switch),
                        );

                        for (idx, arg) in arguments.into_iter().enumerate() {
                            let len = self.instructions.len();
                            let dst = if self.next_reg - 1 < reg + idx + 1 {
                                self.get_next_vreg()
                            } else {
                                reg + idx + 1
                            };
                            assert_eq!(dst, reg + idx + 1);
                            let result = self.eval_node(arg, Some(reg + idx + 1), false);
                            if len == self.instructions.len()
                                || matches!(result, BytecodeResult::And(_) | BytecodeResult::Or(_))
                            {
                                self.add_result(result, dst);
                            }
                            self.fixed_regs.insert(
                                self.instructions.len() - 1,
                                dst.min(self.max_reg - args_len + idx + 1 - self.switch),
                            );
                        }
                        self.switch = 0;
                        self.add_instruction(IROpCode::Call { f_src: reg });
                        BytecodeResult::Reg(reg)
                    }
                    BytecodeResult::Const(Literal::Int(f_idx)) => {
                        let reg = res;
                        let args_len = arguments.len();

                        for (idx, arg) in arguments.into_iter().enumerate() {
                            let len = self.instructions.len();
                            let dst = self.get_next_vreg();
                            assert_eq!(dst, reg + idx + 1);
                            let result = self.eval_node(arg, Some(reg + idx + 1), false);
                            if len == self.instructions.len()
                                || matches!(result, BytecodeResult::And(_) | BytecodeResult::Or(_))
                            {
                                self.add_result(result, dst);
                            }
                            self.fixed_regs.insert(
                                self.instructions.len() - 1,
                                dst.min(self.max_reg - args_len + idx + 1 - self.switch),
                            );
                        }
                        self.switch = 0;
                        self.add_instruction(IROpCode::CallBuiltin {
                            f_idx: f_idx as usize,
                            dst: reg,
                        });
                        self.fixed_regs.insert(
                            self.instructions.len() - 1,
                            reg.min(self.max_reg - args_len - self.switch),
                        );

                        BytecodeResult::Reg(reg)
                    }
                    _ => unreachable!(),
                };

                if let Some(dst_reg) = dst_reg {
                    if dst_reg != res {
                        self.add_instruction(IROpCode::Move {
                            dst: dst_reg,
                            src: res,
                        });
                        return result;
                    }
                }
                result
            }
            ExpressionNode::AnonymousFunction(AnonymousFunctionNode {
                return_type: _,
                parameters,
                body,
            }) => {
                let dst = dst_reg.unwrap_or_else(|| self.get_next_vreg());

                let bp = self.base_pointer;
                self.base_pointer = self.next_reg;
                let params_len = parameters.len();
                self.args_offset += params_len;
                self.next_reg += params_len;

                let func_num = self.functions.len() + 1;
                let identifier = format!("@lambda{}", self.instructions.len());

                assert!(!self.functions.contains_key(&identifier));

                self.functions.insert(identifier.clone(), (func_num, 0, 0));
                self.names.push(HashMap::new());

                self.add_block(self.blocks.len().saturating_sub(1));

                for (idx, (param, _)) in parameters.into_iter().enumerate() {
                    self.names.last_mut().unwrap().insert(param, idx);
                    self.add_instruction(IROpCode::DefParam { dst: idx });
                }
                let len = self.instructions.len();
                self.add_instruction(IROpCode::Nop);
                self.interpret_node(*body, None);
                self.add_instruction(IROpCode::ReturnNil);
                self.instructions[len] = IROpCode::Jmp {
                    delta: (self.instructions.len() - len - 1) as i16,
                };
                self.functions.insert(
                    identifier.clone(),
                    (
                        func_num,
                        self.next_reg - self.base_pointer,
                        len - self.args_offset + 1,
                    ),
                );

                self.reg_num = std::cmp::max(self.reg_num, self.next_reg - self.base_pointer);

                self.next_reg = self.base_pointer;
                self.base_pointer = bp;

                self.names.pop();

                self.add_instruction(IROpCode::LoadFunction {
                    dst,
                    f_idx: self.functions.get(&identifier).unwrap().0 as u16,
                });
                BytecodeResult::Reg(dst)
            }
            ExpressionNode::Identifier(name) => {
                if let Some(index) = self.extern_functions.get(&name) {
                    let index = *index;

                    BytecodeResult::Const(Literal::Int(index as i64))
                } else if let Some(index) = self.names.iter().find_map(|f| f.get(&name)) {
                    BytecodeResult::Reg(*index)
                } else {
                    let reg = dst_reg.unwrap_or_else(|| self.get_next_vreg());
                    self.add_instruction(IROpCode::LoadFunction {
                        dst: reg,
                        f_idx: self.functions.get(&name).unwrap().0 as u16,
                    });
                    BytecodeResult::Reg(reg)
                }
            }
            ExpressionNode::Literal(obj) => match obj {
                ast::Object::Literal(value) => BytecodeResult::Const(value.into()),
                ast::Object::Table(inner) => {
                    let t_dst = dst_reg.unwrap_or_else(|| self.get_next_vreg());
                    self.add_instruction(IROpCode::NewTable {
                        dst: t_dst,
                        len: inner.len(),
                    });
                    for (k, v) in inner {
                        let res = self.eval_node(v, None, false);
                        match k {
                            ast::Literal::Int(k) => {
                                if let Ok(val) = TryInto::<i8>::try_into(k) {
                                    match res {
                                        BytecodeResult::Reg(reg) => {
                                            self.add_instruction(IROpCode::SetIntIndex {
                                                t_dst,
                                                i_idx: val,
                                                val: reg,
                                            });
                                        }
                                        BytecodeResult::Const(l) => {
                                            let c_idx = self.add_constant(l) as u8;
                                            self.add_instruction(IROpCode::SetIntConstIndex {
                                                t_dst,
                                                i_idx: val,
                                                c_idx,
                                            });
                                        }
                                        _ => unreachable!(),
                                    }
                                } else {
                                    match res {
                                        BytecodeResult::Reg(reg) => {
                                            let c_idx = self.add_constant(Literal::Int(k)) as u8;
                                            self.add_instruction(IROpCode::SetConstIndex {
                                                t_dst,
                                                c_idx,
                                                v_src: reg,
                                            });
                                        }
                                        BytecodeResult::Const(l) => {
                                            let c_idx = self.add_constant(l);
                                            let reg = self.get_next_vreg();
                                            self.add_instruction(IROpCode::LoadConst {
                                                dst: reg,
                                                c_idx,
                                            });
                                            let c_idx = self.add_constant(Literal::Int(k)) as u8;
                                            self.add_instruction(IROpCode::SetConstIndex {
                                                t_dst,
                                                c_idx,
                                                v_src: reg,
                                            });
                                        }
                                        _ => unreachable!(),
                                    }
                                }
                            }
                            k => match res {
                                BytecodeResult::Reg(reg) => {
                                    let c_idx = self.add_constant(k.into()) as u8;
                                    self.add_instruction(IROpCode::SetConstIndex {
                                        t_dst,
                                        c_idx,
                                        v_src: reg,
                                    });
                                }
                                BytecodeResult::Const(l) => {
                                    let c_idx = self.add_constant(l);
                                    let reg = self.get_next_vreg();
                                    self.add_instruction(IROpCode::LoadConst { dst: reg, c_idx });
                                    let c_idx = self.add_constant(k.into()) as u8;
                                    self.add_instruction(IROpCode::SetConstIndex {
                                        t_dst,
                                        c_idx,
                                        v_src: reg,
                                    });
                                }
                                _ => unreachable!(),
                            },
                        }
                    }
                    BytecodeResult::Reg(t_dst)
                }
                ast::Object::Nil => {
                    let temp_reg = dst_reg.unwrap_or_else(|| self.get_next_vreg());
                    self.add_instruction(IROpCode::SetNil { dst: temp_reg });
                    BytecodeResult::Reg(temp_reg)
                }
            },
        }
    }

    pub fn build(mut self, allocs: Vec<PReg>) -> Bytecode {
        self.add_instruction(IROpCode::Halt);
        Bytecode {
            extern_fn_names: {
                let mut res = self
                    .extern_functions
                    .into_iter()
                    .collect::<Vec<(String, usize)>>();
                res.sort_by(|a, b| a.1.cmp(&b.1));
                res.into_iter().map(|(name, _)| name).collect()
            },
            functions: {
                let max_reg = allocs.iter().map(|f| f.index()).max().unwrap() + 1;
                let mut res = self
                    .functions
                    .into_iter()
                    .collect::<Vec<(String, (usize, usize, usize))>>();
                res.sort_by(|a, b| a.1 .0.cmp(&b.1 .0));
                let mut res: Vec<(usize, usize)> = res
                    .into_iter()
                    .map(|(_, (_, vregs, addr))| (vregs.min(max_reg), addr))
                    .collect();
                res.insert(0, (self.next_reg.min(max_reg), 0));
                res
            },
            instructions: IROpCode::build_opcode(self.instructions, allocs),
            constants: {
                let mut res = self
                    .constants
                    .into_iter()
                    .collect::<Vec<(Literal, usize)>>();
                res.sort_by(|a, b| a.1.cmp(&b.1));
                res.into_iter().map(|(l, _)| l).collect()
            },
        }
    }

    fn fill_operands(&mut self) {
        self.operands = self
            .instructions
            .iter()
            .enumerate()
            .map(|(idx, op)| match op {
                IROpCode::Halt => vec![],
                IROpCode::Nop => vec![],
                IROpCode::ReturnNil => vec![],
                IROpCode::ReturnConst { .. } => vec![],
                IROpCode::CallBuiltin { f_idx: _, dst } => {
                    vec![if let Some(fixed) = self.fixed_regs.get(&idx) {
                        Operand::reg_fixed_def(
                            VReg::new(*dst, RegClass::Int),
                            PReg::new(*fixed, RegClass::Int),
                        )
                    } else {
                        Operand::any_def(VReg::new(*dst, RegClass::Int))
                    }]
                }
                IROpCode::Jmp { .. } => vec![],
                IROpCode::LoadInt { dst, .. } => {
                    vec![if let Some(fixed) = self.fixed_regs.get(&idx) {
                        Operand::reg_fixed_def(
                            VReg::new(*dst, RegClass::Int),
                            PReg::new(*fixed, RegClass::Int),
                        )
                    } else {
                        Operand::any_def(VReg::new(*dst, RegClass::Int))
                    }]
                }
                IROpCode::LoadBool { dst, .. } => {
                    vec![if let Some(fixed) = self.fixed_regs.get(&idx) {
                        Operand::reg_fixed_def(
                            VReg::new(*dst, RegClass::Int),
                            PReg::new(*fixed, RegClass::Int),
                        )
                    } else {
                        Operand::any_def(VReg::new(*dst, RegClass::Int))
                    }]
                }
                IROpCode::LoadConst { dst, .. } => {
                    vec![if let Some(fixed) = self.fixed_regs.get(&idx) {
                        Operand::reg_fixed_def(
                            VReg::new(*dst, RegClass::Int),
                            PReg::new(*fixed, RegClass::Int),
                        )
                    } else {
                        Operand::any_def(VReg::new(*dst, RegClass::Int))
                    }]
                }
                IROpCode::Move { dst, src } => vec![
                    if let Some(fixed) = self.fixed_regs.get(&idx) {
                        Operand::reg_fixed_def(
                            VReg::new(*dst, RegClass::Int),
                            PReg::new(*fixed, RegClass::Int),
                        )
                    } else {
                        Operand::any_def(VReg::new(*dst, RegClass::Int))
                    },
                    Operand::any_use(VReg::new(*src, RegClass::Int)),
                ],
                IROpCode::Add { lhs, rhs, dst } => vec![
                    Operand::any_use(VReg::new(*lhs, RegClass::Int)),
                    Operand::any_use(VReg::new(*rhs, RegClass::Int)),
                    if let Some(fixed) = self.fixed_regs.get(&idx) {
                        Operand::reg_fixed_def(
                            VReg::new(*dst, RegClass::Int),
                            PReg::new(*fixed, RegClass::Int),
                        )
                    } else {
                        Operand::any_def(VReg::new(*dst, RegClass::Int))
                    },
                ],
                IROpCode::AddConst { src, dst, .. } => vec![
                    Operand::any_use(VReg::new(*src, RegClass::Int)),
                    if let Some(fixed) = self.fixed_regs.get(&idx) {
                        Operand::reg_fixed_def(
                            VReg::new(*dst, RegClass::Int),
                            PReg::new(*fixed, RegClass::Int),
                        )
                    } else {
                        Operand::any_def(VReg::new(*dst, RegClass::Int))
                    },
                ],
                IROpCode::Sub { lhs, rhs, dst } => vec![
                    Operand::any_use(VReg::new(*lhs, RegClass::Int)),
                    Operand::any_use(VReg::new(*rhs, RegClass::Int)),
                    if let Some(fixed) = self.fixed_regs.get(&idx) {
                        Operand::reg_fixed_def(
                            VReg::new(*dst, RegClass::Int),
                            PReg::new(*fixed, RegClass::Int),
                        )
                    } else {
                        Operand::any_def(VReg::new(*dst, RegClass::Int))
                    },
                ],
                IROpCode::Mul { lhs, rhs, dst } => vec![
                    Operand::any_use(VReg::new(*lhs, RegClass::Int)),
                    Operand::any_use(VReg::new(*rhs, RegClass::Int)),
                    if let Some(fixed) = self.fixed_regs.get(&idx) {
                        Operand::reg_fixed_def(
                            VReg::new(*dst, RegClass::Int),
                            PReg::new(*fixed, RegClass::Int),
                        )
                    } else {
                        Operand::any_def(VReg::new(*dst, RegClass::Int))
                    },
                ],
                IROpCode::MulConst { src, dst, .. } => vec![
                    Operand::any_use(VReg::new(*src, RegClass::Int)),
                    if let Some(fixed) = self.fixed_regs.get(&idx) {
                        Operand::reg_fixed_def(
                            VReg::new(*dst, RegClass::Int),
                            PReg::new(*fixed, RegClass::Int),
                        )
                    } else {
                        Operand::any_def(VReg::new(*dst, RegClass::Int))
                    },
                ],
                IROpCode::Div { lhs, rhs, dst } => vec![
                    Operand::any_use(VReg::new(*lhs, RegClass::Int)),
                    Operand::any_use(VReg::new(*rhs, RegClass::Int)),
                    if let Some(fixed) = self.fixed_regs.get(&idx) {
                        Operand::reg_fixed_def(
                            VReg::new(*dst, RegClass::Int),
                            PReg::new(*fixed, RegClass::Int),
                        )
                    } else {
                        Operand::any_def(VReg::new(*dst, RegClass::Int))
                    },
                ],
                IROpCode::DivConst { src, dst, .. } => vec![
                    Operand::any_use(VReg::new(*src, RegClass::Int)),
                    if let Some(fixed) = self.fixed_regs.get(&idx) {
                        Operand::reg_fixed_def(
                            VReg::new(*dst, RegClass::Int),
                            PReg::new(*fixed, RegClass::Int),
                        )
                    } else {
                        Operand::any_def(VReg::new(*dst, RegClass::Int))
                    },
                ],
                IROpCode::Mod { lhs, rhs, dst } => vec![
                    Operand::any_use(VReg::new(*lhs, RegClass::Int)),
                    Operand::any_use(VReg::new(*rhs, RegClass::Int)),
                    if let Some(fixed) = self.fixed_regs.get(&idx) {
                        Operand::reg_fixed_def(
                            VReg::new(*dst, RegClass::Int),
                            PReg::new(*fixed, RegClass::Int),
                        )
                    } else {
                        Operand::any_def(VReg::new(*dst, RegClass::Int))
                    },
                ],
                IROpCode::Pow { lhs, rhs, dst } => vec![
                    Operand::any_use(VReg::new(*lhs, RegClass::Int)),
                    Operand::any_use(VReg::new(*rhs, RegClass::Int)),
                    if let Some(fixed) = self.fixed_regs.get(&idx) {
                        Operand::reg_fixed_def(
                            VReg::new(*dst, RegClass::Int),
                            PReg::new(*fixed, RegClass::Int),
                        )
                    } else {
                        Operand::any_def(VReg::new(*dst, RegClass::Int))
                    },
                ],
                IROpCode::IntDiv { lhs, rhs, dst } => vec![
                    Operand::any_use(VReg::new(*lhs, RegClass::Int)),
                    Operand::any_use(VReg::new(*rhs, RegClass::Int)),
                    if let Some(fixed) = self.fixed_regs.get(&idx) {
                        Operand::reg_fixed_def(
                            VReg::new(*dst, RegClass::Int),
                            PReg::new(*fixed, RegClass::Int),
                        )
                    } else {
                        Operand::any_def(VReg::new(*dst, RegClass::Int))
                    },
                ],
                IROpCode::Not { src, dst } => vec![
                    Operand::any_use(VReg::new(*src, RegClass::Int)),
                    if let Some(fixed) = self.fixed_regs.get(&idx) {
                        Operand::reg_fixed_def(
                            VReg::new(*dst, RegClass::Int),
                            PReg::new(*fixed, RegClass::Int),
                        )
                    } else {
                        Operand::any_def(VReg::new(*dst, RegClass::Int))
                    },
                ],
                IROpCode::UnaryMinus { src, dst } => vec![
                    Operand::any_use(VReg::new(*src, RegClass::Int)),
                    if let Some(fixed) = self.fixed_regs.get(&idx) {
                        Operand::reg_fixed_def(
                            VReg::new(*dst, RegClass::Int),
                            PReg::new(*fixed, RegClass::Int),
                        )
                    } else {
                        Operand::any_def(VReg::new(*dst, RegClass::Int))
                    },
                ],
                IROpCode::SetNil { dst } => vec![if let Some(fixed) = self.fixed_regs.get(&idx) {
                    Operand::reg_fixed_def(
                        VReg::new(*dst, RegClass::Int),
                        PReg::new(*fixed, RegClass::Int),
                    )
                } else {
                    Operand::any_def(VReg::new(*dst, RegClass::Int))
                }],
                IROpCode::NewTable { dst, .. } => {
                    vec![if let Some(fixed) = self.fixed_regs.get(&idx) {
                        Operand::reg_fixed_def(
                            VReg::new(*dst, RegClass::Int),
                            PReg::new(*fixed, RegClass::Int),
                        )
                    } else {
                        Operand::any_def(VReg::new(*dst, RegClass::Int))
                    }]
                }
                IROpCode::SetIndex {
                    t_dst,
                    i_src,
                    v_src,
                } => vec![
                    Operand::any_use(VReg::new(*t_dst, RegClass::Int)),
                    Operand::any_use(VReg::new(*i_src, RegClass::Int)),
                    Operand::any_use(VReg::new(*v_src, RegClass::Int)),
                ],
                IROpCode::GetIndex { dst, t_src, v_src } => vec![
                    if let Some(fixed) = self.fixed_regs.get(&idx) {
                        Operand::reg_fixed_def(
                            VReg::new(*dst, RegClass::Int),
                            PReg::new(*fixed, RegClass::Int),
                        )
                    } else {
                        Operand::any_def(VReg::new(*dst, RegClass::Int))
                    },
                    Operand::any_use(VReg::new(*t_src, RegClass::Int)),
                    Operand::any_use(VReg::new(*v_src, RegClass::Int)),
                ],
                IROpCode::Call { f_src, .. } => {
                    vec![Operand::any_use(VReg::new(*f_src, RegClass::Int))]
                }
                IROpCode::Return { src } => vec![Operand::any_use(VReg::new(*src, RegClass::Int))],
                IROpCode::Test { src, .. } => {
                    vec![Operand::any_use(VReg::new(*src, RegClass::Int))]
                }
                IROpCode::Eq { lhs, rhs, .. } => vec![
                    Operand::any_use(VReg::new(*lhs, RegClass::Int)),
                    Operand::any_use(VReg::new(*rhs, RegClass::Int)),
                ],
                IROpCode::Lt { lhs, rhs, .. } => vec![
                    Operand::any_use(VReg::new(*lhs, RegClass::Int)),
                    Operand::any_use(VReg::new(*rhs, RegClass::Int)),
                ],
                IROpCode::Lte { lhs, rhs, .. } => vec![
                    Operand::any_use(VReg::new(*lhs, RegClass::Int)),
                    Operand::any_use(VReg::new(*rhs, RegClass::Int)),
                ],
                IROpCode::EqConst { src, .. }
                | IROpCode::LtConst { src, .. }
                | IROpCode::LteConst { src, .. }
                | IROpCode::GtConst { src, .. }
                | IROpCode::GteConst { src, .. }
                | IROpCode::EqInt { src, .. }
                | IROpCode::LtInt { src, .. }
                | IROpCode::LteInt { src, .. }
                | IROpCode::GtInt { src, .. }
                | IROpCode::GteInt { src, .. } => {
                    vec![Operand::any_use(VReg::new(*src, RegClass::Int))]
                }
                IROpCode::SetConstIndex {
                    t_dst,
                    c_idx: _,
                    v_src,
                } => vec![
                    Operand::any_use(VReg::new(*t_dst, RegClass::Int)),
                    Operand::any_use(VReg::new(*v_src, RegClass::Int)),
                ],
                IROpCode::SetIntIndex {
                    t_dst,
                    i_idx: _,
                    val,
                } => vec![
                    Operand::any_use(VReg::new(*t_dst, RegClass::Int)),
                    Operand::any_use(VReg::new(*val, RegClass::Int)),
                ],
                IROpCode::SetIntConstIndex { t_dst, .. } => {
                    vec![Operand::any_use(VReg::new(*t_dst, RegClass::Int))]
                }
                IROpCode::SetIndexConstVal { t_dst, i_src, .. } => vec![
                    Operand::any_use(VReg::new(*t_dst, RegClass::Int)),
                    Operand::any_use(VReg::new(*i_src, RegClass::Int)),
                ],
                IROpCode::GetIntIndex { dst, t_src, .. } => vec![
                    if let Some(fixed) = self.fixed_regs.get(&idx) {
                        Operand::reg_fixed_def(
                            VReg::new(*dst, RegClass::Int),
                            PReg::new(*fixed, RegClass::Int),
                        )
                    } else {
                        Operand::any_def(VReg::new(*dst, RegClass::Int))
                    },
                    Operand::any_use(VReg::new(*t_src, RegClass::Int)),
                ],
                IROpCode::GetConstIndex { dst, t_src, .. } => vec![
                    if let Some(fixed) = self.fixed_regs.get(&idx) {
                        Operand::reg_fixed_def(
                            VReg::new(*dst, RegClass::Int),
                            PReg::new(*fixed, RegClass::Int),
                        )
                    } else {
                        Operand::any_def(VReg::new(*dst, RegClass::Int))
                    },
                    Operand::any_use(VReg::new(*t_src, RegClass::Int)),
                ],
                IROpCode::LoadFunction { dst, .. } => {
                    vec![if let Some(fixed) = self.fixed_regs.get(&idx) {
                        Operand::reg_fixed_def(
                            VReg::new(*dst, RegClass::Int),
                            PReg::new(*fixed, RegClass::Int),
                        )
                    } else {
                        Operand::any_def(VReg::new(*dst, RegClass::Int))
                    }]
                }
                IROpCode::AddInt { src, dst, .. } => vec![
                    Operand::any_use(VReg::new(*src, RegClass::Int)),
                    if let Some(fixed) = self.fixed_regs.get(&idx) {
                        Operand::reg_fixed_def(
                            VReg::new(*dst, RegClass::Int),
                            PReg::new(*fixed, RegClass::Int),
                        )
                    } else {
                        Operand::any_def(VReg::new(*dst, RegClass::Int))
                    },
                ],
                IROpCode::DefParam { dst } => {
                    vec![Operand::any_def(VReg::new(*dst, RegClass::Int))]
                }
                IROpCode::ForLoop { src, dst } => vec![
                    Operand::any_use(VReg::new(*src, RegClass::Int)),
                    Operand::any_def(VReg::new(*dst, RegClass::Int)),
                ],
                IROpCode::TestNil { src } => vec![Operand::any_use(VReg::new(*src, RegClass::Int))],
                IROpCode::ModConst { lhs, c_idx: _, dst } => vec![
                    Operand::any_use(VReg::new(*lhs, RegClass::Int)),
                    Operand::any_def(VReg::new(*dst, RegClass::Int)),
                ],
            })
            .collect();
    }
}

impl CompileRegister for BytecodeBuilder {
    type Output = BytecodeBuilder;

    fn from_ast(ast: Node, max_reg: usize) -> Self::Output {
        let mut bytecode = BytecodeBuilder {
            max_reg,
            ..Default::default()
        };
        bytecode.add_block(0);
        bytecode.interpret_node(ast, None);
        bytecode.end_block();
        bytecode.reg_num = std::cmp::max(bytecode.reg_num, bytecode.next_reg);
        bytecode.fill_operands();
        bytecode
    }
}
