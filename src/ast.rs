use std::{collections::HashMap, fmt};

use ordered_float::OrderedFloat;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BinaryOperator {
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    Add,
    Substract,
    Divide,
    Multiply,
    Power,
    Mod,
    Div,
    LogicalAnd,
    LogicalOr,
}

#[derive(Debug, Clone)]
pub enum Object {
    Literal(Literal),
    Table(HashMap<Literal, ExpressionNode>),
    Nil,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Literal {
    Int(i64),
    Float(OrderedFloat<f64>),
    Str(String),
    Bool(bool),
}

#[derive(Debug, Clone, Eq)]
pub enum Type {
    Int,
    Float,
    Str,
    Bool,
    Table(Vec<(Literal, TypeAnnotation)>),
    Function(Vec<TypeAnnotation>, Option<TypeAnnotation>),
    Nil,
    Any,
}

impl ::core::cmp::PartialEq for Type {
    #[inline]
    fn eq(&self, other: &Type) -> bool {
        let self_tag = ::core::mem::discriminant(self);
        let other_tag = ::core::mem::discriminant(other);
        self_tag == other_tag
            && match (self, other) {
                (Type::Function(params1, ret1), Type::Function(params2, ret2)) => {
                    *params1 == *params2 && *ret1 == *ret2
                }
                _ => true,
            }
    }
}

impl ::core::hash::Hash for Type {
    fn hash<H: ::core::hash::Hasher>(&self, state: &mut H) {
        let __self_tag = ::core::mem::discriminant(self);
        ::core::hash::Hash::hash(&__self_tag, state);
        if let Type::Function(params, ret) = self {
            ::core::hash::Hash::hash(params, state);
            ::core::hash::Hash::hash(ret, state)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeAnnotation(pub Vec<Type>);

impl fmt::Display for TypeAnnotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        let types = self.0.iter().fold(String::new(), |acc, v| {
            if acc.is_empty() {
                v.to_string()
            } else {
                acc + " | " + &v.to_string()
            }
        });
        write!(f, "{types}")
    }
}

impl TypeAnnotation {
    pub fn find_duplicates(&self, other: &Self) -> Self {
        let mut hashmap = HashMap::new();
        for x in self.0.iter().chain(other.0.iter()) {
            *hashmap.entry(x.clone()).or_insert(0) += 1;
        }
        TypeAnnotation(
            hashmap
                .into_iter()
                .filter(|(_, v)| *v > 1)
                .map(|(k, _)| k)
                .collect(),
        )
    }
}
impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match &self {
            Type::Int => write!(f, "int"),
            Type::Nil => write!(f, "nil"),
            Type::Float => write!(f, "float"),
            Type::Str => write!(f, "str"),
            Type::Bool => write!(f, "bool"),
            Type::Table(fields) => {
                let fields = fields
                    .iter()
                    .fold(String::new(), |acc, (field, field_type)| {
                        if acc.is_empty() {
                            format!("{field}: {field_type}")
                        } else {
                            acc + ", " + &format!("{field}: {field_type}")
                        }
                    });
                write!(f, "table[{fields}]")
            }
            Type::Function(params, return_type) => {
                let params = params
                    .iter()
                    .map(|param_type| format!("{param_type}"))
                    .fold(String::new(), |acc, v| {
                        if acc.is_empty() {
                            v
                        } else {
                            acc + ", " + &v
                        }
                    });
                if let Some(ret_type) = return_type {
                    write!(f, "fn({params}) -> {ret_type}")
                } else {
                    write!(f, "fn({params})")
                }
            }
            Type::Any => write!(f, "any"),
        }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match &self {
            Literal::Int(n) => write!(f, "{n}"),
            Literal::Float(n) => write!(f, "{n}"),
            Literal::Str(s) => write!(f, "{s}"),
            Literal::Bool(b) => write!(f, "{b}"),
        }
    }
}

impl fmt::Display for Object {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match &self {
            Object::Literal(l) => write!(f, "{l}"),
            Object::Nil => write!(f, "nil"),
            Object::Table(t) => {
                let params =
                    t.iter()
                        .map(|(k, v)| format!("{k}: {v}"))
                        .fold(String::new(), |acc, v| {
                            if acc.is_empty() {
                                v
                            } else {
                                acc + ", " + &v
                            }
                        });
                write!(f, "{{ {params} }}")
            }
        }
    }
}

impl fmt::Display for BinaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match &self {
            BinaryOperator::Add => write!(f, "+"),
            BinaryOperator::Substract => write!(f, "-"),
            BinaryOperator::Divide => write!(f, "/"),
            BinaryOperator::Multiply => write!(f, "*"),
            BinaryOperator::Power => write!(f, "^"),
            BinaryOperator::Mod => write!(f, "%"),
            BinaryOperator::Div => write!(f, "\\"),
            BinaryOperator::Equal => write!(f, "=="),
            BinaryOperator::NotEqual => write!(f, "!="),
            BinaryOperator::LessThan => write!(f, "<"),
            BinaryOperator::LessThanOrEqual => write!(f, "<="),
            BinaryOperator::GreaterThan => write!(f, ">"),
            BinaryOperator::GreaterThanOrEqual => write!(f, ">="),
            BinaryOperator::LogicalAnd => write!(f, "&&"),
            BinaryOperator::LogicalOr => write!(f, "||"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Node {
    Program(Vec<Node>),
    VariableDeclaration(VariableDeclarationNode),
    Assignment(AssignmentNode),
    WhileStatement(WhileLoopNode),
    IfStatement(IfStatementNode),
    ForeachStatement(ForeachLoopNode),
    FunctionDeclaration(FunctionDeclarationNode),
    ExternFunctionDeclaration(ExternFunctionDeclarationNode),
    Expression(ExpressionNode),
    StatementBlock(Box<Node>),
    ReturnStatement(Option<Box<ExpressionNode>>),
    BreakStatement,
    ContinueStatement,
}
#[derive(Debug, Clone, PartialEq)]
pub struct TypeAliasNode {
    pub identifier: String,
    pub alias: TypeAnnotation,
}

#[derive(Debug, Clone)]
pub struct FunctionDeclarationNode {
    pub return_type: Option<TypeAnnotation>,
    pub identifier: String,
    pub parameters: Vec<(String, TypeAnnotation)>,
    pub body: Box<Node>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExternFunctionDeclarationNode {
    pub return_type: Option<TypeAnnotation>,
    pub identifier: String,
    pub parameters: Vec<(String, TypeAnnotation)>,
}

#[derive(Debug, Clone)]
pub struct IfStatementNode {
    pub condition: Box<ExpressionNode>,
    pub if_block: Box<Node>,
    pub else_block: Option<Box<Node>>,
}

#[derive(Debug, Clone)]
pub struct VariableDeclarationNode {
    pub datatype: TypeAnnotation,
    pub identifier: String,
    pub initializer: Box<ExpressionNode>,
}

#[derive(Debug, Clone)]
pub struct WhileLoopNode {
    pub condition: Box<ExpressionNode>,
    pub body: Box<Node>,
}

#[derive(Debug, Clone)]
pub struct ForeachLoopNode {
    pub identifier: String,
    pub table: Box<ExpressionNode>,
    pub body: Box<Node>,
}

#[derive(Debug, Clone)]
pub struct AssignmentNode {
    pub identifier: Box<ExpressionNode>,
    pub expression: Box<ExpressionNode>,
}

#[derive(Debug, Clone)]
pub enum ExpressionNode {
    BinaryOperation(Box<ExpressionNode>, BinaryOperator, Box<ExpressionNode>),
    UnaryOperation(UnaryOperator, Box<ExpressionNode>),
    FunctionCall(FunctionCallNode),
    AnonymousFunction(AnonymousFunctionNode),
    Identifier(String),
    Literal(Object),
}

#[derive(Debug, Clone)]
pub struct FunctionCallNode {
    pub identifier: Box<ExpressionNode>,
    pub arguments: Vec<ExpressionNode>,
}

#[derive(Debug, Clone)]
pub struct AnonymousFunctionNode {
    pub return_type: Option<TypeAnnotation>,
    pub parameters: Vec<(String, TypeAnnotation)>,
    pub body: Box<Node>,
}

#[derive(Debug, Clone)]
pub enum UnaryOperator {
    Negate,
    LogicalNot,
    Index(Box<ExpressionNode>),
}

impl fmt::Display for UnaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            UnaryOperator::Negate => write!(f, "-"),
            UnaryOperator::LogicalNot => write!(f, "!"),
            UnaryOperator::Index(e) => write!(f, "[{e}]"),
        }
    }
}

impl fmt::Display for AnonymousFunctionNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        let params = self
            .parameters
            .iter()
            .map(|(ident, param_type)| format!("{ident}: {param_type}"))
            .fold(String::new(), |acc, v| {
                if acc.is_empty() {
                    v
                } else {
                    acc + ", " + &v
                }
            });
        match &self.return_type {
            Some(ret) => write!(f, "fn({}) -> {} {}", params, ret, self.body),
            None => write!(f, "fn({}) {}", params, self.body),
        }
    }
}

impl fmt::Display for FunctionCallNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        let params = self.arguments.iter().fold(String::new(), |acc, v| {
            if acc.is_empty() {
                v.to_string()
            } else {
                acc + "," + &v.to_string()
            }
        });
        write!(f, "{}({})", self.identifier, params)
    }
}

impl fmt::Display for TypeAliasNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "type {} = {}", self.identifier, self.alias)
    }
}

impl fmt::Display for FunctionDeclarationNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        let params = self
            .parameters
            .iter()
            .map(|(ident, param_type)| format!("{ident}: {param_type}"))
            .fold(String::new(), |acc, v| {
                if acc.is_empty() {
                    v
                } else {
                    acc + ", " + &v
                }
            });
        match &self.return_type {
            Some(ret) => write!(
                f,
                "fn {}({}) -> {} {}",
                self.identifier, params, ret, self.body
            ),
            None => write!(f, "fn {}({}) {}", self.identifier, params, self.body),
        }
    }
}

impl fmt::Display for ExternFunctionDeclarationNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        let params = self
            .parameters
            .iter()
            .map(|(ident, param_type)| format!("{ident}:{param_type}"))
            .fold(String::new(), |acc, v| {
                if acc.is_empty() {
                    v
                } else {
                    acc + "," + &v
                }
            });
        match &self.return_type {
            Some(ret) => write!(f, "extern fn {}({}) -> {}", self.identifier, params, ret),
            None => write!(f, "extern fn {}({})", self.identifier, params),
        }
    }
}

impl fmt::Display for IfStatementNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "if {} {}", self.condition, self.if_block)?;
        if let Some(else_block) = &self.else_block {
            write!(f, "else {}", else_block)?;
        }
        Ok(())
    }
}

impl fmt::Display for WhileLoopNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "while {} {}", self.condition, self.body)
    }
}

impl fmt::Display for ForeachLoopNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "for {} in {} {}", self.identifier, self.table, self.body)
    }
}

impl fmt::Display for AssignmentNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{} = {}", self.identifier, self.expression)
    }
}

impl fmt::Display for VariableDeclarationNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(
            f,
            "var {}: {} = {}",
            self.identifier, self.datatype, self.initializer
        )
    }
}

impl fmt::Display for ExpressionNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            ExpressionNode::BinaryOperation(lhs, op, rhs) => write!(f, "{lhs} {op} {rhs}"),
            ExpressionNode::UnaryOperation(op, rhs) => match op {
                UnaryOperator::Negate => write!(f, "{op}({rhs})"),
                UnaryOperator::LogicalNot => write!(f, "{op}({rhs})"),
                UnaryOperator::Index(_) => write!(f, "({rhs}){op}"),
            },
            ExpressionNode::FunctionCall(node) => write!(f, "{node}"),
            ExpressionNode::AnonymousFunction(node) => write!(f, "{node}"),
            ExpressionNode::Identifier(s) => write!(f, "{s}"),
            ExpressionNode::Literal(o) => write!(f, "{o}"),
        }
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            Node::Program(lines) => {
                let lines = lines.iter().fold(String::new(), |acc, node| {
                    if !acc.is_empty() {
                        acc + "\n" + &node.to_string()
                    } else {
                        node.to_string()
                    }
                });
                write!(f, "{lines}")
            }
            Node::ForeachStatement(node) => write!(f, "{node}"),
            Node::VariableDeclaration(node) => write!(f, "{node}"),
            Node::Assignment(node) => write!(f, "{node}"),
            Node::WhileStatement(node) => write!(f, "{node}"),
            Node::IfStatement(node) => write!(f, "{node}"),
            Node::FunctionDeclaration(node) => write!(f, "{node}"),
            Node::ExternFunctionDeclaration(node) => write!(f, "{node}"),
            Node::Expression(node) => write!(f, "{node}"),
            Node::StatementBlock(node) => write!(f, "{{\n{node}\n}}"),
            Node::ReturnStatement(node) => {
                if let Some(node) = node {
                    write!(f, "return {node}")
                } else {
                    write!(f, "return")
                }
            }
            Node::BreakStatement => write!(f, "break"),
            Node::ContinueStatement => write!(f, "continue"),
        }
    }
}
