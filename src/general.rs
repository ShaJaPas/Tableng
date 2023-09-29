use std::{collections::HashMap, fmt::Display};

use ordered_float::OrderedFloat;

use crate::ast::{self, AnonymousFunctionNode};

#[derive(Debug, Clone, PartialEq)]
pub enum Object {
    Literal(Literal),
    Table(HashMap<Literal, usize>),
    Function(AnonymousFunctionNode),
    ExternFunction(String),
    Nil,
}

impl PartialEq for AnonymousFunctionNode {
    fn eq(&self, other: &Self) -> bool {
        self.return_type == other.return_type && self.parameters == other.parameters
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Literal {
    Int(i64),
    Float(OrderedFloat<f64>),
    Str(String),
    Bool(bool),
}

impl Display for Literal {
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
            ast::Literal::Str(v) => Self::Str(v),
            ast::Literal::Bool(v) => Self::Bool(v),
        }
    }
}
