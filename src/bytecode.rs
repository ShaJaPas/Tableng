use hashbrown::HashMap;
use ordered_float::OrderedFloat;

use crate::{
    ast::{
        self, AnonymousFunctionNode, AssignmentNode, ExpressionNode, ExternFunctionDeclarationNode,
        ForeachLoopNode, FunctionCallNode, FunctionDeclarationNode, IfStatementNode, Node,
        VariableDeclarationNode, WhileLoopNode,
    },
    opcode::OpCode,
    CompileStack,
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

#[derive(PartialEq, Debug)]
pub struct BytecodeBuilder {
    names: HashMap<String, usize>,
    constants: Vec<Literal>,
    instructions: Vec<OpCode>,
    extern_functions: HashMap<String, usize>,
    loop_start: usize,
    local_names: Vec<HashMap<String, usize>>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct Bytecode {
    pub names: Vec<String>,
    pub extern_fn_names: Vec<String>,
    pub constants: Vec<Literal>,
    pub instructions: Vec<OpCode>,
}

impl Bytecode {
    pub fn empty() -> Self {
        Self {
            names: Vec::new(),
            constants: Vec::new(),
            instructions: Vec::new(),
            extern_fn_names: Vec::new(),
        }
    }
}
impl Default for BytecodeBuilder {
    fn default() -> Self {
        Self {
            names: HashMap::new(),
            extern_functions: HashMap::new(),
            constants: vec!["__next", "__eq", "__neq", "__gt", "__gte", "__lt", "__lte"]
                .into_iter()
                .map(|f| Literal::Str(Box::new(f.to_string())))
                .collect(),
            instructions: Vec::new(),
            loop_start: 0,
            local_names: Vec::new(),
        }
    }
}

impl BytecodeBuilder {
    fn add_instruction(&mut self, op_code: OpCode) {
        self.instructions.push(op_code);
    }

    pub fn interpret_node(&mut self, node: Node) {
        match node {
            Node::Program(statments) => {
                for stmt in statments {
                    self.interpret_node(stmt);
                }
            }
            Node::VariableDeclaration(VariableDeclarationNode {
                datatype: _, // TODO: runtime table type checking
                identifier,
                initializer,
            }) => {
                self.eval_node(*initializer);
                self.names
                    .try_insert(identifier.clone(), self.names.len())
                    .ok();
                let index = *self.names.get(&identifier).unwrap();
                self.add_instruction(OpCode::MakeVariable(index as u32));
            }
            Node::Assignment(AssignmentNode {
                identifier,
                expression,
            }) => {
                let ident = *identifier;
                self.eval_node(*expression);
                match ident {
                    ExpressionNode::UnaryOperation(op, rhs) => {
                        self.eval_node(*rhs);
                        match op {
                            ast::UnaryOperator::Index(exp) => {
                                self.eval_node(*exp);
                                self.add_instruction(OpCode::StoreIndex);
                            }
                            _ => unreachable!(),
                        }
                    }
                    ExpressionNode::Identifier(name) => {
                        if let Some(Some(local)) = self.local_names.last().map(|f| f.get(&name)) {
                            self.add_instruction(OpCode::StoreLocalVariable(*local as u32));
                        } else if let Some(index) = self.names.get(&name) {
                            self.add_instruction(OpCode::StoreVariable(*index as u32));
                        }
                    }
                    _ => unreachable!(),
                }
            }
            Node::WhileStatement(WhileLoopNode { condition, body }) => {
                let ip_start = self.instructions.len().saturating_sub(1);
                self.eval_node(*condition);
                self.add_instruction(OpCode::JumpForwardIfFalse(0));
                let ip = self.instructions.len() - 1;
                self.loop_start = ip;
                self.interpret_node(*body);
                self.instructions.extend_from_within(ip_start..ip);
                self.add_instruction(OpCode::JumpBackwardIfTrue(
                    (self.instructions.len() - 1 - ip) as u32,
                ));
                // Replace all NOPs by jump forward as it is break statement
                for i in self.instructions[self.loop_start..]
                    .iter()
                    .enumerate()
                    .filter(|(_, f)| matches!(f, OpCode::Nop))
                    .map(|(i, _)| i)
                    .collect::<Vec<usize>>()
                {
                    self.instructions[self.loop_start + i] =
                        OpCode::JumpForward((self.instructions.len() - self.loop_start - i) as u32);
                }
                self.instructions[ip] =
                    OpCode::JumpForwardIfFalse((self.instructions.len() - 1 - ip) as u32);
            }
            Node::IfStatement(IfStatementNode {
                condition,
                if_block,
                else_block,
            }) => {
                self.eval_node(*condition);
                self.add_instruction(OpCode::JumpForwardIfFalse(0));
                let ip = self.instructions.len() - 1;
                self.interpret_node(*if_block);
                self.instructions[ip] =
                    OpCode::JumpForwardIfFalse((self.instructions.len() - 1 - ip) as u32);
                if let Some(else_block) = else_block {
                    self.interpret_node(*else_block);
                }
            }
            Node::ForeachStatement(ForeachLoopNode {
                identifier,
                table,
                body,
            }) => {
                self.names
                    .try_insert(identifier.clone(), self.names.len())
                    .ok();
                let var_i = *self.names.get(&identifier).unwrap();
                self.add_instruction(OpCode::PushNil);
                self.add_instruction(OpCode::MakeVariable(var_i as u32));
                self.eval_node(*table);
                self.add_instruction(OpCode::GetNext(0));
                let ip = self.instructions.len() - 1;
                self.loop_start = ip;
                //self.add_instruction(OpCode::StoreVariable(var_i));
                self.interpret_node(*body);
                // Replace all NOPs by jump forward as it is break statement
                for i in self.instructions[self.loop_start..]
                    .iter()
                    .enumerate()
                    .filter(|(_, f)| matches!(f, OpCode::Nop))
                    .map(|(i, _)| i)
                    .collect::<Vec<usize>>()
                {
                    self.instructions[self.loop_start + i] =
                        OpCode::JumpForward((self.instructions.len() - self.loop_start - i) as u32);
                }
                let delta = self.instructions.len() - ip;
                self.add_instruction(OpCode::JumpBackward(delta as u32 + 1));
                self.instructions[ip] = OpCode::GetNext(delta as u32);
            }
            Node::FunctionDeclaration(FunctionDeclarationNode {
                return_type: _,
                identifier,
                parameters,
                body,
            }) => {
                self.names
                    .try_insert(identifier.clone(), self.names.len())
                    .ok();
                let var_i = *self.names.get(&identifier).unwrap();
                self.local_names.push(HashMap::new());
                for (i, (name, _)) in parameters.into_iter().rev().enumerate() {
                    self.local_names.last_mut().unwrap().insert(name, i);
                }
                self.add_instruction(OpCode::MakeFunction(self.instructions.len() as u32 + 3));
                self.add_instruction(OpCode::MakeVariable(var_i as u32));
                self.add_instruction(OpCode::JumpForward(0));
                let ip = self.instructions.len() - 1;
                self.interpret_node(*body);
                self.add_instruction(OpCode::PushNil);
                self.add_instruction(OpCode::Return);
                let delta = self.instructions.len() - 1 - ip;
                self.instructions[ip] = OpCode::JumpForward(delta as u32);
                self.local_names.pop();
            }
            Node::ExternFunctionDeclaration(ExternFunctionDeclarationNode {
                return_type: _,
                identifier,
                parameters: _,
            }) => {
                self.extern_functions
                    .try_insert(identifier.clone(), self.extern_functions.len())
                    .ok();
            }
            Node::Expression(expr) => self.eval_node(expr),
            Node::StatementBlock(stmt) => self.interpret_node(*stmt),
            Node::ReturnStatement(ret) => {
                if let Some(ret) = ret {
                    self.eval_node(*ret);
                } else {
                    self.add_instruction(OpCode::PushNil);
                }
                self.add_instruction(OpCode::Return);
            }
            Node::BreakStatement => self.add_instruction(OpCode::Nop),
            Node::ContinueStatement => self.add_instruction(OpCode::JumpBackward(
                (self.instructions.len() - self.loop_start + 1) as u32,
            )),
        }
    }

    pub fn eval_node(&mut self, node: ExpressionNode) {
        match node {
            ExpressionNode::BinaryOperation(lhs, op, rhs) => {
                self.eval_node(*lhs);
                self.eval_node(*rhs);
                match op {
                    ast::BinaryOperator::Equal => self.add_instruction(OpCode::BinaryEquals),
                    ast::BinaryOperator::NotEqual => self.add_instruction(OpCode::BinaryNotEquals),
                    ast::BinaryOperator::LessThan => self.add_instruction(OpCode::BinaryLessThan),
                    ast::BinaryOperator::LessThanOrEqual => {
                        self.add_instruction(OpCode::BinaryLessThanOrEquals)
                    }
                    ast::BinaryOperator::GreaterThan => {
                        self.add_instruction(OpCode::BinaryGreaterThan)
                    }
                    ast::BinaryOperator::GreaterThanOrEqual => {
                        self.add_instruction(OpCode::BinaryGreaterThanOrEquals)
                    }
                    ast::BinaryOperator::Add => self.add_instruction(OpCode::BinaryPlus),
                    ast::BinaryOperator::Substract => self.add_instruction(OpCode::BinaryMinus),
                    ast::BinaryOperator::Divide => self.add_instruction(OpCode::BinaryDivision),
                    ast::BinaryOperator::Multiply => {
                        self.add_instruction(OpCode::BinaryMultiplication)
                    }
                    ast::BinaryOperator::Power => self.add_instruction(OpCode::BinaryPower),
                    ast::BinaryOperator::Mod => self.add_instruction(OpCode::BinaryMod),
                    ast::BinaryOperator::Div => self.add_instruction(OpCode::BinaryDiv),
                    ast::BinaryOperator::LogicalAnd => self.add_instruction(OpCode::BinaryAnd),
                    ast::BinaryOperator::LogicalOr => self.add_instruction(OpCode::BinaryOr),
                }
            }
            ExpressionNode::UnaryOperation(op, rhs) => {
                self.eval_node(*rhs);
                match op {
                    ast::UnaryOperator::Negate => self.add_instruction(OpCode::UnaryMinus),
                    ast::UnaryOperator::LogicalNot => self.add_instruction(OpCode::UnaryNot),
                    ast::UnaryOperator::Index(exp) => {
                        self.eval_node(*exp);
                        self.add_instruction(OpCode::Index);
                    }
                }
            }
            ExpressionNode::FunctionCall(FunctionCallNode {
                identifier,
                arguments,
            }) => {
                let len = arguments.len();
                for arg in arguments.into_iter().rev() {
                    self.eval_node(arg);
                }
                let idx = if let ExpressionNode::Identifier(name) = &*identifier {
                    self.extern_functions.get(name).copied()
                } else {
                    None
                };
                self.eval_node(*identifier);
                self.add_instruction(if let Some(idx) = idx {
                    OpCode::CallBuiltin(idx as u32)
                } else {
                    OpCode::Call(len as u32)
                });
            }
            ExpressionNode::AnonymousFunction(AnonymousFunctionNode {
                return_type: _,
                parameters,
                body,
            }) => {
                self.local_names.push(HashMap::new());
                for (i, (name, _)) in parameters.into_iter().rev().enumerate() {
                    self.local_names.last_mut().unwrap().insert(name, i);
                }
                self.add_instruction(OpCode::MakeFunction(self.instructions.len() as u32 + 2));
                self.add_instruction(OpCode::JumpForward(0));
                let ip = self.instructions.len() - 1;
                self.interpret_node(*body);
                self.add_instruction(OpCode::PushNil);
                self.add_instruction(OpCode::Return);
                let delta = self.instructions.len() - 1 - ip;
                self.instructions[ip] = OpCode::JumpForward(delta as u32);
                self.local_names.pop();
            }
            ExpressionNode::Identifier(name) => {
                if let Some(Some(local)) = self.local_names.last().map(|f| f.get(&name)) {
                    self.add_instruction(OpCode::PushLocalVariable(*local as u32));
                } else if let Some(index) = self.names.get(&name) {
                    self.add_instruction(OpCode::PushVariable(*index as u32));
                }
            }
            ExpressionNode::Literal(obj) => match obj {
                ast::Object::Literal(value) => {
                    self.constants.push(value.into());
                    self.add_instruction(OpCode::PushConst(self.constants.len() as u32 - 1));
                }
                ast::Object::Table(inner) => {
                    let len = inner.len();
                    for (k, v) in inner {
                        self.eval_node(v);
                        self.constants.push(k.into());
                        self.add_instruction(OpCode::PushConst(self.constants.len() as u32 - 1));
                    }
                    self.add_instruction(OpCode::MakeTable(len as u32));
                }
                ast::Object::Nil => self.add_instruction(OpCode::PushNil),
            },
        }
    }

    pub fn build(mut self) -> Bytecode {
        self.add_instruction(OpCode::Halt);
        Bytecode {
            names: {
                let mut res = self.names.into_iter().collect::<Vec<(String, usize)>>();
                res.sort_by(|a, b| a.1.cmp(&b.1));
                res.into_iter().map(|(name, _)| name).collect()
            },
            extern_fn_names: {
                let mut res = self
                    .extern_functions
                    .into_iter()
                    .collect::<Vec<(String, usize)>>();
                res.sort_by(|a, b| a.1.cmp(&b.1));
                res.into_iter().map(|(name, _)| name).collect()
            },
            constants: self.constants,
            instructions: self.instructions,
        }
    }
}

impl CompileStack for BytecodeBuilder {
    type Output = Bytecode;

    fn from_ast(ast: Node) -> Self::Output {
        let mut bytecode = BytecodeBuilder::default();
        bytecode.interpret_node(ast);
        bytecode.build()
    }
}
