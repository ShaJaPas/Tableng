pub mod ast;
pub mod bytecode;
pub mod general;
pub mod ir;
pub mod ir_gen;
pub mod opcode;
pub mod opcode_v2;
pub mod parser;
pub mod reg_alloc;
pub mod vm;
pub mod vm_v2;

use ast::Node;
use parser::{Rule, TablengParser};
use pest::error::Error;

#[derive(thiserror::Error, Debug)]
pub enum ParseError {
    #[error("Syntax error:\n{0}")]
    SyntaxError(#[from] Box<Error<Rule>>),
    #[error("Runtime error: {message}")]
    RuntimeError { message: String },
}

pub trait CompileStack {
    type Output;

    fn from_ast(ast: Node) -> Self::Output;

    fn from_source(source: &str, file_path: Option<String>) -> Result<Self::Output, ParseError> {
        let parser = TablengParser::default();
        Ok(Self::from_ast(parser.parse(source, file_path)?))
    }
}

pub trait CompileRegister {
    type Output;

    fn from_ast(ast: Node, max_reg: usize) -> Self::Output;

    fn from_source(
        source: &str,
        file_path: Option<String>,
        max_reg: usize,
    ) -> Result<Self::Output, ParseError> {
        let parser = TablengParser::default();
        Ok(Self::from_ast(parser.parse(source, file_path)?, max_reg))
    }
}
