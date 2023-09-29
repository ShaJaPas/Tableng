use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fs::File,
    io::Read,
    path::Path,
};

use pest::{
    error::{Error, ErrorVariant},
    iterators::{Pair, Pairs},
    pratt_parser::{Assoc::Left, Op, PrattParser},
    Parser, Position,
};

use crate::{ast::*, ParseError};

#[derive(pest_derive::Parser)]
#[grammar = "syntax.pest"]
struct InnnerParser;

type Function = RefCell<Vec<HashMap<String, (Vec<TypeAnnotation>, Option<TypeAnnotation>)>>>;
type Alias = RefCell<Vec<HashMap<String, Vec<Type>>>>;
type Variable = RefCell<Vec<HashMap<String, TypeAnnotation>>>;
struct SymbolTable {
    variables: Variable,
    type_aliases: Alias,
    functions: Function,
    current_fn_scope: RefCell<Vec<Option<TypeAnnotation>>>,
}

impl SymbolTable {
    fn new() -> Self {
        SymbolTable {
            variables: RefCell::new(vec![HashMap::new()]),
            type_aliases: RefCell::new(vec![HashMap::new()]),
            functions: RefCell::new(vec![HashMap::new()]),
            current_fn_scope: RefCell::new(vec![]),
        }
    }

    fn enter_scope(&self) {
        self.variables.borrow_mut().push(HashMap::new());
        self.type_aliases.borrow_mut().push(HashMap::new());
        self.functions.borrow_mut().push(HashMap::new());
    }

    fn exit_scope(&self) {
        self.variables.borrow_mut().pop();
        self.type_aliases.borrow_mut().pop();
        self.functions.borrow_mut().pop();
    }

    fn add_variable(&self, name: String, var_type: TypeAnnotation) {
        let mut variables = self.variables.borrow_mut();
        let current_scope = variables.last_mut().unwrap();
        current_scope.insert(name, var_type);
    }

    fn find_variable(&self, name: &str) -> Option<TypeAnnotation> {
        for scope in self.variables.borrow().iter().rev() {
            if let Some(symbol) = scope.get(name) {
                return Some(symbol.clone());
            }
        }
        None
    }

    fn add_function(
        &self,
        name: String,
        args: Vec<TypeAnnotation>,
        ret_type: Option<TypeAnnotation>,
    ) {
        let mut functions = self.functions.borrow_mut();
        let current_scope = functions.last_mut().unwrap();
        current_scope.insert(name, (args, ret_type));
    }

    fn find_function(&self, name: &str) -> Option<(Vec<TypeAnnotation>, Option<TypeAnnotation>)> {
        for scope in self.functions.borrow().iter().rev() {
            if let Some(symbol) = scope.get(name) {
                return Some(symbol.clone());
            }
        }
        None
    }

    fn add_type_alias(&self, name: String, var_type: Vec<Type>) {
        let mut symbols = self.type_aliases.borrow_mut();
        let current_scope = symbols.last_mut().unwrap();
        current_scope.insert(name, var_type);
    }

    fn find_type_alias(&self, name: &str) -> Option<Vec<Type>> {
        for scope in self.type_aliases.borrow().iter().rev() {
            if let Some(types) = scope.get(name) {
                return Some(types.clone());
            }
        }
        None
    }
}

pub struct TablengParser {
    pratt: PrattParser<Rule>,
    symbols: SymbolTable,
    file_path: RefCell<Option<String>>,
}

impl Default for TablengParser {
    fn default() -> Self {
        Self {
            pratt: PrattParser::new()
                .op(Op::infix(Rule::logicalOr, Left))
                .op(Op::infix(Rule::logicalAnd, Left))
                .op(Op::infix(Rule::ComparisonOperator, Left))
                .op(Op::infix(Rule::Add, Left) | Op::infix(Rule::Subtract, Left))
                .op(Op::infix(Rule::Multiply, Left)
                    | Op::infix(Rule::Divide, Left)
                    | Op::infix(Rule::Mod, Left)
                    | Op::infix(Rule::Div, Left))
                .op(Op::infix(Rule::Power, Left))
                .op(Op::prefix(Rule::logicalNot))
                .op(Op::prefix(Rule::unaryMinus))
                .op(Op::postfix(Rule::index)),
            symbols: SymbolTable::new(),
            file_path: RefCell::new(None),
        }
    }
}
impl TablengParser {
    pub fn parse(&self, source: &str, file_path: Option<String>) -> Result<Node, ParseError> {
        *self.file_path.borrow_mut() = file_path;
        let pairs = InnnerParser::parse(Rule::program, source)
            .map_err(|err| Into::<ParseError>::into(Box::new(err)))?;
        let mut statements = Vec::new();
        for pair in pairs {
            if pair.as_rule() != Rule::EOI {
                statements.push(self.build_ast(pair)?);
            }
        }
        Ok(Node::Program(statements))
    }

    fn build_ast(&self, pair: Pair<Rule>) -> Result<Node, ParseError> {
        Ok(match pair.as_rule() {
            Rule::variableDeclaration => self.build_variable_declaration(pair.into_inner())?,
            Rule::expression => Node::Expression(self.build_expression(pair.into_inner())?.0),
            Rule::assignment => self.build_assignment(pair.into_inner())?,
            Rule::whileStatement => self.build_while_loop(pair.into_inner())?,
            Rule::statementBlock => {
                self.symbols.enter_scope();
                let mut statements = Vec::new();
                for pair in pair.into_inner() {
                    if pair.as_rule() != Rule::EOI {
                        statements.push(self.build_ast(pair)?);
                    }
                }
                self.symbols.exit_scope();
                Node::StatementBlock(Box::new(Node::Program(statements)))
            }
            Rule::ifStatement => self.build_if_statement(pair.into_inner())?,
            Rule::forStatement => self.build_for_loop(pair.into_inner())?,
            Rule::foreachStatement => self.build_foreach_loop(pair.into_inner())?,
            Rule::returnStatement => self.build_return_statement(pair)?,
            Rule::functionDeclaration => self.build_fn_declaration(pair.into_inner())?,
            Rule::externFunctionDeclaration => {
                self.build_extern_fn_declaration(pair.into_inner())?
            }
            Rule::breakStatement => Node::BreakStatement,
            Rule::continueStatement => Node::ContinueStatement,
            Rule::typeDefinition => {
                let mut inner = pair.into_inner();
                let ident = inner.next().unwrap().as_str().trim().to_string();
                let type_alias = self.get_type(inner.next().unwrap())?;
                self.symbols.add_type_alias(ident, type_alias);
                Node::Program(Vec::new())
            }
            Rule::importStatement => {
                let path = pair.into_inner().next().unwrap();
                let curr_path = self.file_path.borrow().clone().unwrap_or(String::new());
                let current_path = Path::new(&curr_path).parent().unwrap();
                let file_path =
                    current_path.join(format!("{}.tabl", path.as_str().trim_matches('"')));
                let str_path = file_path.to_str().unwrap().to_string();
                match File::open(file_path) {
                    Ok(mut file) => {
                        let mut buf = String::new();
                        file.read_to_string(&mut buf).unwrap();
                        self.parse(&buf, Some(str_path))?
                    }
                    Err(err) => {
                        return Err(Into::<ParseError>::into(Box::new(
                            Error::new_from_pos(
                                ErrorVariant::CustomError {
                                    message: format!("Cannot import file: {}", err),
                                },
                                Position::new(
                                    path.get_input(),
                                    path.get_input()
                                        .lines()
                                        .take(path.line_col().0 - 1)
                                        .map(|s| s.len() + 1)
                                        .sum::<usize>()
                                        + path.line_col().1
                                        - 1,
                                )
                                .unwrap(),
                            )
                            .with_path(&self.file_path.borrow().clone().unwrap_or(String::new())),
                        )))
                    }
                }
            }
            rule => unreachable!("Unexpected rule: {rule:?}"),
        })
    }

    fn build_assignment(&self, mut inner_pairs: Pairs<Rule>) -> Result<Node, ParseError> {
        let source = inner_pairs.get_input();
        let ident_input = inner_pairs.next().unwrap();
        let ident_str = ident_input.as_str().trim();
        let identifier = self.build_expression(ident_input.into_inner())?;
        let exp_input = inner_pairs.next().unwrap();
        let exp_str = exp_input.as_str().trim();
        let (line, col) = exp_input.line_col();
        let expression = self.build_expression(exp_input.into_inner())?;

        if !identifier.1 .0.iter().any(|f| matches!(f, Type::Any))
            && !expression.1 .0.iter().any(|f| matches!(f, Type::Any))
            && identifier.1.find_duplicates(&expression.1).0.is_empty()
        {
            return Err(Into::<ParseError>::into(Box::new(
                Error::new_from_pos(
                    ErrorVariant::CustomError {
                        message: format!(
                            "Type mismatch: `{}` has type \"{}\", while `{}` has type \"{}\"",
                            ident_str, identifier.1, exp_str, expression.1
                        ),
                    },
                    Position::new(
                        source,
                        source
                            .lines()
                            .take(line - 1)
                            .map(|s| s.len() + 1)
                            .sum::<usize>()
                            + col
                            - 1,
                    )
                    .unwrap(),
                )
                .with_path(&self.file_path.borrow().clone().unwrap_or(String::new())),
            )));
        }
        Ok(Node::Assignment(AssignmentNode {
            identifier: Box::new(identifier.0),
            expression: Box::new(expression.0),
        }))
    }

    fn build_fn_declaration(&self, mut inner_pairs: Pairs<Rule>) -> Result<Node, ParseError> {
        let fn_name = inner_pairs.next().unwrap();
        let identifier = fn_name.as_str().trim().to_string();
        let args = inner_pairs.next().unwrap();
        let parameters = args
            .into_inner()
            .map(|pair| {
                let mut inner = pair.into_inner();
                let ident = inner.next().unwrap().as_str().to_string();
                let arg_type = self.get_type(inner.next().unwrap())?;
                Ok((ident, arg_type))
            })
            .collect::<Result<Vec<(String, Vec<Type>)>, ParseError>>()?;

        let parameters: Vec<(String, TypeAnnotation)> = parameters
            .into_iter()
            .map(|(name, typ)| (name, TypeAnnotation(typ)))
            .collect();

        let expression = inner_pairs.next().unwrap();

        if self
            .symbols
            .functions
            .borrow()
            .last()
            .unwrap()
            .contains_key(&identifier)
        {
            return Err(Into::<ParseError>::into(Box::new(
                Error::new_from_pos(
                    ErrorVariant::CustomError {
                        message: "Function is already defined".to_string(),
                    },
                    Position::new(
                        fn_name.get_input(),
                        fn_name
                            .get_input()
                            .lines()
                            .take(fn_name.line_col().0 - 1)
                            .map(|s| s.len() + 1)
                            .sum::<usize>()
                            + fn_name.line_col().1
                            - 1,
                    )
                    .unwrap(),
                )
                .with_path(&self.file_path.borrow().clone().unwrap_or(String::new())),
            )));
        }
        let (return_type, body) = if expression.as_rule() == Rule::functionReturn {
            let ret_type = Some(TypeAnnotation(
                self.get_type(expression.into_inner().next().unwrap())?,
            ));

            self.symbols.add_function(
                identifier.clone(),
                parameters
                    .iter()
                    .map(|(_, arg_type)| arg_type.clone())
                    .collect(),
                ret_type.clone(),
            );
            self.symbols.enter_scope();
            self.symbols
                .current_fn_scope
                .borrow_mut()
                .push(ret_type.clone());
            parameters.iter().for_each(|(ident, arg_type)| {
                self.symbols
                    .add_variable(ident.trim().to_string(), arg_type.clone())
            });

            (ret_type, self.build_ast(inner_pairs.next().unwrap())?)
        } else {
            self.symbols.add_function(
                identifier.clone(),
                parameters
                    .iter()
                    .map(|(_, arg_type)| arg_type.clone())
                    .collect(),
                None,
            );
            self.symbols.enter_scope();
            self.symbols.current_fn_scope.borrow_mut().push(None);
            parameters.iter().for_each(|(ident, arg_type)| {
                self.symbols
                    .add_variable(ident.trim().to_string(), arg_type.clone())
            });

            (None, self.build_ast(expression)?)
        };
        self.symbols.current_fn_scope.borrow_mut().pop();

        self.symbols.exit_scope();

        Ok(Node::FunctionDeclaration(FunctionDeclarationNode {
            return_type,
            identifier,
            parameters,
            body: Box::new(body),
        }))
    }

    fn build_extern_fn_declaration(
        &self,
        mut inner_pairs: Pairs<Rule>,
    ) -> Result<Node, ParseError> {
        let identifier = inner_pairs.next().unwrap().as_str().trim().to_string();
        let args = inner_pairs.next().unwrap();
        let parameters = args
            .into_inner()
            .map(|pair| {
                let mut inner = pair.into_inner();
                Ok((
                    inner.next().unwrap().as_str().to_string(),
                    self.get_type(inner.next().unwrap())?,
                ))
            })
            .collect::<Result<Vec<(String, Vec<Type>)>, ParseError>>()?;

        let parameters: Vec<(String, TypeAnnotation)> = parameters
            .into_iter()
            .map(|(name, typ)| (name, TypeAnnotation(typ)))
            .collect();

        let return_type = inner_pairs.next().map(|f| {
            Ok(TypeAnnotation(
                self.get_type(f.into_inner().next().unwrap())?,
            ))
        });

        if let Some(Err(err)) = return_type {
            return Err(err);
        }
        let return_type = return_type.map(|f| f.unwrap());

        self.symbols.add_function(
            identifier.clone(),
            parameters
                .iter()
                .map(|(_, arg_type)| arg_type.clone())
                .collect(),
            return_type.clone(),
        );
        self.symbols.enter_scope();
        parameters.iter().for_each(|(ident, arg_type)| {
            self.symbols
                .add_variable(ident.trim().to_string(), arg_type.clone())
        });
        self.symbols.exit_scope();
        Ok(Node::ExternFunctionDeclaration(
            ExternFunctionDeclarationNode {
                return_type,
                identifier,
                parameters,
            },
        ))
    }

    fn build_return_statement(&self, pair: Pair<Rule>) -> Result<Node, ParseError> {
        let mut inner_pairs = pair.clone().into_inner();
        let exp = inner_pairs.next();
        let expression = exp.map(|pair| self.build_expression(pair.into_inner()));

        if let Some(Err(e)) = expression {
            return Err(e);
        }
        let expression = expression.map(|f| f.unwrap());

        if let Some(ret_type) = self.symbols.current_fn_scope.borrow().last() {
            if !match (ret_type, &expression) {
                (None, None) => true,
                (Some(t), Some((_, typ))) => {
                    t.0.iter().any(|f| matches!(f, Type::Any))
                        || typ.0.iter().any(|f| matches!(f, Type::Any))
                        || !t.find_duplicates(typ).0.is_empty()
                }
                _ => false,
            } {
                return Err(Into::<ParseError>::into(Box::new(Error::new_from_pos(
                    ErrorVariant::CustomError {
                        message: match (ret_type, &expression) {
                            (None, Some((_, ret))) => format!("Type mismatch: Function has no return type, while return statement has type \"{}\"",
                            ret
                        ),
                        (Some(ret), None) => format!("Type mismatch: Function returns \"{}\", while return statement is empty",
                            ret
                        ),
                        (Some(ret), Some((_, ret_typ))) => format!("Type mismatch: Function returns \"{}\", while return statement has type \"{}\"",
                            ret, ret_typ
                        ),
                        _ => unreachable!()
                        }
                    },
                    Position::new(
                        pair.get_input(),
                        pair
                            .get_input()
                            .lines()
                            .take(pair.line_col().0 - 1)
                            .map(|s| s.len() + 1)
                            .sum::<usize>()
                            + pair.line_col().1
                            - 1,
                    )
                    .unwrap(),
                ).with_path(&self.file_path.borrow().clone().unwrap_or(String::new())))));
            }
        } else {
            return Err(Into::<ParseError>::into(Box::new(
                Error::new_from_pos(
                    ErrorVariant::CustomError {
                        message: "Cannot use return statement outside of a function".to_string(),
                    },
                    Position::new(
                        pair.get_input(),
                        pair.get_input()
                            .lines()
                            .take(pair.line_col().0 - 1)
                            .map(|s| s.len() + 1)
                            .sum::<usize>()
                            + pair.line_col().1
                            - 1,
                    )
                    .unwrap(),
                )
                .with_path(&self.file_path.borrow().clone().unwrap_or(String::new())),
            )));
        }
        Ok(Node::ReturnStatement(expression.map(|f| Box::new(f.0))))
    }

    fn build_if_statement(&self, mut inner_pairs: Pairs<Rule>) -> Result<Node, ParseError> {
        let cond_pair = inner_pairs.next().unwrap();
        let condition = self.build_expression(cond_pair.clone().into_inner())?;

        if !condition
            .1
             .0
            .iter()
            .any(|f| matches!(f, Type::Any | Type::Bool))
        {
            return Err(Into::<ParseError>::into(Box::new(
                Error::new_from_pos(
                    ErrorVariant::CustomError {
                        message: format!("Type mismatch: Expected `bool`, got `{}`", condition.1),
                    },
                    Position::new(
                        cond_pair.get_input(),
                        cond_pair
                            .get_input()
                            .lines()
                            .take(cond_pair.line_col().0 - 1)
                            .map(|s| s.len() + 1)
                            .sum::<usize>()
                            + cond_pair.line_col().1
                            - 1,
                    )
                    .unwrap(),
                )
                .with_path(&self.file_path.borrow().clone().unwrap_or(String::new())),
            )));
        }
        let statements = self.build_ast(inner_pairs.next().unwrap())?;
        let else_block = inner_pairs.next().map(|pair| self.build_ast(pair));

        if let Some(Err(e)) = else_block {
            return Err(e);
        }
        let else_block = else_block.map(|f| Box::new(Node::Program(vec![f.unwrap()])));
        Ok(Node::IfStatement(IfStatementNode {
            condition: Box::new(condition.0),
            if_block: Box::new(statements),
            else_block,
        }))
    }

    fn build_while_loop(&self, mut inner_pairs: Pairs<Rule>) -> Result<Node, ParseError> {
        let cond_pair = inner_pairs.next().unwrap();
        let condition = self.build_expression(cond_pair.clone().into_inner())?;

        if !condition
            .1
             .0
            .iter()
            .any(|f| matches!(f, Type::Any | Type::Bool))
        {
            return Err(Into::<ParseError>::into(Box::new(
                Error::new_from_pos(
                    ErrorVariant::CustomError {
                        message: format!("Type mismatch: Expected `bool`, got `{}`", condition.1),
                    },
                    Position::new(
                        cond_pair.get_input(),
                        cond_pair
                            .get_input()
                            .lines()
                            .take(cond_pair.line_col().0 - 1)
                            .map(|s| s.len() + 1)
                            .sum::<usize>()
                            + cond_pair.line_col().1
                            - 1,
                    )
                    .unwrap(),
                )
                .with_path(&self.file_path.borrow().clone().unwrap_or(String::new())),
            )));
        }

        let statements = self.build_ast(inner_pairs.next().unwrap())?;
        Ok(Node::WhileStatement(WhileLoopNode {
            condition: Box::new(condition.0),
            body: Box::new(statements),
        }))
    }

    fn build_for_loop(&self, mut inner_pairs: Pairs<Rule>) -> Result<Node, ParseError> {
        let ident_input = inner_pairs.next().unwrap();
        let identifier = ident_input.as_str().trim().to_string();
        let start_exp = inner_pairs.next().unwrap();
        let start = self.build_expression(start_exp.clone().into_inner())?;
        let end_exp = inner_pairs.next().unwrap();
        let end = self.build_expression(end_exp.clone().into_inner())?;

        for (exp, pair) in [(&start, &start_exp), (&end, &end_exp)] {
            if !exp.1 .0.iter().any(|f| matches!(f, Type::Any | Type::Int)) {
                return Err(Into::<ParseError>::into(Box::new(
                    Error::new_from_pos(
                        ErrorVariant::CustomError {
                            message: format!("Type mismatch: Expected `int`, got `{}`", exp.1),
                        },
                        Position::new(
                            pair.get_input(),
                            pair.get_input()
                                .lines()
                                .take(pair.line_col().0 - 1)
                                .map(|s| s.len() + 1)
                                .sum::<usize>()
                                + pair.line_col().1
                                - 1,
                        )
                        .unwrap(),
                    )
                    .with_path(&self.file_path.borrow().clone().unwrap_or(String::new())),
                )));
            }
        }

        let body = inner_pairs.next().unwrap();
        let source = &format!(
            "
        foreach {} in {{
            \"start\": {},
            \"i\": {},
            \"end\": {},
            \"__next\": fn(self: table[]) -> int {{
                var step = 1
                if self[\"end\"] < self[\"start\"] {{
                    step = -1
                }}
                if self[\"i\"] != self[\"end\"] + step {{
                    var res = self[\"i\"]
                    self[\"i\"] = self[\"i\"] + step
                    return res
                }}
            }}
        }} {{{}}}",
            identifier,
            start_exp.as_str(),
            start_exp.as_str(),
            end_exp.as_str(),
            body.as_str()
        );
        let pairs = InnnerParser::parse(Rule::program, source)
            .unwrap()
            .next()
            .unwrap()
            .into_inner();
        self.build_foreach_loop(pairs)
    }

    fn build_foreach_loop(&self, mut inner_pairs: Pairs<Rule>) -> Result<Node, ParseError> {
        let ident_input = inner_pairs.next().unwrap();
        let identifier = ident_input.as_str().trim().to_string();
        let table_exp = inner_pairs.next().unwrap();
        let table = self.build_expression(table_exp.clone().into_inner())?;

        if !table
            .1
             .0
            .iter()
            .any(|f| matches!(f, Type::Any | Type::Table(..)))
        {
            return Err(Into::<ParseError>::into(Box::new(
                Error::new_from_pos(
                    ErrorVariant::CustomError {
                        message: format!("Type mismatch: Expected `table`, got `{}`", table.1),
                    },
                    Position::new(
                        table_exp.get_input(),
                        table_exp
                            .get_input()
                            .lines()
                            .take(table_exp.line_col().0 - 1)
                            .map(|s| s.len() + 1)
                            .sum::<usize>()
                            + table_exp.line_col().1
                            - 1,
                    )
                    .unwrap(),
                )
                .with_path(&self.file_path.borrow().clone().unwrap_or(String::new())),
            )));
        }

        self.symbols.enter_scope();
        self.symbols
            .add_variable(identifier.clone(), TypeAnnotation(vec![Type::Any]));
        let statements = self.build_ast(inner_pairs.next().unwrap())?;
        self.symbols.exit_scope();

        Ok(Node::ForeachStatement(ForeachLoopNode {
            identifier,
            table: Box::new(table.0),
            body: Box::new(statements),
        }))
    }

    fn build_variable_declaration(&self, mut inner_pairs: Pairs<Rule>) -> Result<Node, ParseError> {
        let ident_input = inner_pairs.next().unwrap();
        let identifier = ident_input.as_str().trim().to_string();
        let datatype_input = inner_pairs.next().unwrap();
        let (datatype, initializer) = if datatype_input.as_rule() == Rule::TypeAnnotation {
            (
                TypeAnnotation(self.get_type(datatype_input.clone())?),
                self.build_expression(inner_pairs.clone())?,
            )
        } else {
            let exp = self.build_expression(datatype_input.clone().into_inner())?;
            (exp.1.clone(), exp)
        };

        if !datatype.0.iter().any(|f| matches!(f, Type::Any))
            && datatype.find_duplicates(&initializer.1).0.is_empty()
        {
            return Err(Into::<ParseError>::into(Box::new(
                Error::new_from_pos(
                    ErrorVariant::CustomError {
                        message: format!(
                            "Type mismatch: `{}` has type \"{}\", while `{}` has type \"{}\"",
                            &identifier,
                            datatype,
                            inner_pairs.as_str().trim(),
                            initializer.1
                        ),
                    },
                    Position::new(
                        ident_input.get_input(),
                        ident_input
                            .get_input()
                            .lines()
                            .take(ident_input.line_col().0 - 1)
                            .map(|s| s.len() + 1)
                            .sum::<usize>()
                            + ident_input.line_col().1
                            - 1,
                    )
                    .unwrap(),
                )
                .with_path(&self.file_path.borrow().clone().unwrap_or(String::new())),
            )));
        }

        let variable_declaration_node = VariableDeclarationNode {
            identifier: identifier.clone(),
            datatype,
            initializer: Box::new(initializer.0),
        };

        self.symbols
            .add_variable(identifier, variable_declaration_node.datatype.clone());
        Ok(Node::VariableDeclaration(variable_declaration_node))
    }

    fn get_type(&self, datatype: Pair<Rule>) -> Result<Vec<Type>, ParseError> {
        let mut result = HashSet::new();
        datatype.clone().into_inner().try_for_each(|f| {
            match f.as_rule() {
                Rule::TableDef => {
                    let fields = f
                        .into_inner()
                        .map(|pair| {
                            let mut inner = pair.into_inner();
                            let literal_node = inner.next().unwrap().into_inner().next().unwrap();
                            Ok((
                                match literal_node.as_rule() {
                                    Rule::Int => {
                                        Literal::Int(literal_node.as_str().parse().unwrap())
                                    }
                                    Rule::Float => {
                                        Literal::Float(literal_node.as_str().parse().unwrap())
                                    }
                                    Rule::Bool => {
                                        Literal::Bool(literal_node.as_str().parse().unwrap())
                                    }
                                    Rule::String => Literal::Str(
                                        literal_node.as_str().trim_matches('"').to_string(),
                                    ),
                                    _ => unreachable!(),
                                },
                                self.get_type(inner.next().unwrap())?,
                            ))
                        })
                        .collect::<Result<Vec<(Literal, Vec<Type>)>, ParseError>>()?;

                    let fields: Vec<(Literal, TypeAnnotation)> = fields
                        .into_iter()
                        .map(|(name, typ)| (name, TypeAnnotation(typ)))
                        .collect();
                    result.insert(Type::Table(fields));
                }
                Rule::anonymousFunctionType => {
                    let mut inner_pairs = f.into_inner();
                    let args = inner_pairs.next().unwrap();
                    let parameters = args
                        .into_inner()
                        .map(|pair| self.get_type(pair))
                        .collect::<Result<Vec<Vec<Type>>, ParseError>>()?;

                    let parameters: Vec<TypeAnnotation> =
                        parameters.into_iter().map(TypeAnnotation).collect();

                    let return_type = inner_pairs
                        .next()
                        .map(|f| self.get_type(f.into_inner().next().unwrap()));
                    if let Some(Err(e)) = return_type {
                        return Err(e);
                    }
                    let return_type = return_type.map(|f| TypeAnnotation(f.unwrap()));

                    result.insert(Type::Function(parameters, return_type));
                }
                Rule::FloatType => {
                    result.insert(Type::Float);
                }
                Rule::IntType => {
                    result.insert(Type::Int);
                }
                Rule::StringType => {
                    result.insert(Type::Str);
                }
                Rule::BoolType => {
                    result.insert(Type::Bool);
                }
                Rule::AnyType => {
                    result.insert(Type::Any);
                }
                Rule::identifier => {
                    match self.symbols.find_type_alias(f.as_str().trim()) {
                        Some(x) => {
                            x.into_iter().for_each(|f| {
                                result.insert(f);
                            });
                        }
                        None => {
                            return Err(Into::<ParseError>::into(Box::new(
                                Error::new_from_pos(
                                    ErrorVariant::CustomError {
                                        message: format!(
                                            "Cannot find type `{}` in this scope",
                                            f.as_str()
                                        ),
                                    },
                                    Position::new(
                                        f.get_input(),
                                        f.get_input()
                                            .lines()
                                            .take(f.line_col().0 - 1)
                                            .map(|s| s.len() + 1)
                                            .sum::<usize>()
                                            + f.line_col().1
                                            - 1,
                                    )
                                    .unwrap(),
                                )
                                .with_path(
                                    &self.file_path.borrow().clone().unwrap_or(String::new()),
                                ),
                            )));
                        }
                    };
                }
                rule => unreachable!("{rule:?}"),
            };
            Ok(())
        })?;
        Ok(result.into_iter().collect())
    }

    fn build_expression(
        &self,
        pairs: Pairs<Rule>,
    ) -> Result<(ExpressionNode, TypeAnnotation), ParseError> {
        self.pratt
            .map_primary(|primary| match primary.as_rule() {
                Rule::identifier => {
                    let identifier = primary.as_str().to_string();
                    if let Some(ident) = self.symbols.find_variable(&identifier) {
                        Ok((ExpressionNode::Identifier(identifier), ident))
                    } else if let Some((params, ret_type)) = self.symbols.find_function(&identifier)
                    {
                        Ok((
                            ExpressionNode::Identifier(identifier),
                            TypeAnnotation(vec![Type::Function(params, ret_type)]),
                        ))
                    } else {
                        return Err(Box::new(
                            Error::new_from_pos(
                                ErrorVariant::CustomError {
                                    message: format!(
                                        "Cannot find variable `{identifier}` in this scope"
                                    ),
                                },
                                Position::new(
                                    primary.get_input(),
                                    primary
                                        .get_input()
                                        .lines()
                                        .take(primary.line_col().0 - 1)
                                        .map(|s| s.len() + 1)
                                        .sum::<usize>()
                                        + primary.line_col().1
                                        - 1,
                                )
                                .unwrap(),
                            )
                            .with_path(&self.file_path.borrow().clone().unwrap_or(String::new())),
                        )
                        .into());
                    }
                }
                Rule::literal => {
                    let literal_node = primary.into_inner().next().unwrap();
                    let (literal, object_type) = match literal_node.as_rule() {
                        Rule::Int => (
                            Object::Literal(Literal::Int(literal_node.as_str().parse().unwrap())),
                            TypeAnnotation(vec![Type::Int]),
                        ),
                        Rule::Float => (
                            Object::Literal(Literal::Float(literal_node.as_str().parse().unwrap())),
                            TypeAnnotation(vec![Type::Float]),
                        ),
                        Rule::Bool => (
                            Object::Literal(Literal::Bool(literal_node.as_str().parse().unwrap())),
                            TypeAnnotation(vec![Type::Bool]),
                        ),
                        Rule::String => (
                            Object::Literal(Literal::Str(
                                literal_node.as_str().trim_matches('"').to_string(),
                            )),
                            TypeAnnotation(vec![Type::Str]),
                        ),
                        _ => unreachable!(),
                    };
                    Ok((ExpressionNode::Literal(literal), object_type))
                }
                Rule::functionCall => {
                    let mut inner = primary.into_inner();
                    let prefix_expression = inner.next().unwrap();
                    let arguments = inner.next();
                    let (exp, typ) =
                        self.build_expression(prefix_expression.clone().into_inner())?;
                    Ok(if typ.0.iter().any(|f| matches!(f, Type::Function(..))) {
                        if let Some(Type::Function(params, ret_type)) =
                            typ.0.into_iter().find(|f| matches!(f, Type::Function(..)))
                        {
                            (
                                ExpressionNode::FunctionCall(FunctionCallNode {
                                    identifier: Box::new(exp),
                                    arguments: if let Some(args) = arguments {
                                        let args = args
                                            .into_inner()
                                            .map(|exp| self.build_expression(exp.into_inner()))
                                            .collect::<Result<
                                                Vec<(ExpressionNode, TypeAnnotation)>,
                                                ParseError,
                                            >>()?;
                                        if args.len() != params.len()
                                            || !params.iter().enumerate().all(|(i, typ)| {
                                                typ.0.iter().any(|f| matches!(f, Type::Any))
                                                    || !params[i].find_duplicates(typ).0.is_empty()
                                            })
                                        {
                                            return Err(Box::new(
                                                Error::new_from_pos(
                                                    ErrorVariant::CustomError {
                                                        message: "Incompatible parameters type"
                                                            .to_string(),
                                                    },
                                                    Position::new(
                                                        prefix_expression.get_input(),
                                                        prefix_expression
                                                            .get_input()
                                                            .lines()
                                                            .take(
                                                                prefix_expression.line_col().0 - 1,
                                                            )
                                                            .map(|s| s.len() + 1)
                                                            .sum::<usize>()
                                                            + prefix_expression.line_col().1
                                                            - 1,
                                                    )
                                                    .unwrap(),
                                                )
                                                .with_path(
                                                    &self
                                                        .file_path
                                                        .borrow()
                                                        .clone()
                                                        .unwrap_or(String::new()),
                                                ),
                                            )
                                            .into());
                                        }
                                        args.into_iter().map(|f| f.0).collect()
                                    } else {
                                        if !params.is_empty() {
                                            return Err(Box::new(
                                                Error::new_from_pos(
                                                    ErrorVariant::CustomError {
                                                        message: "Incompatible parameters type"
                                                            .to_string(),
                                                    },
                                                    Position::new(
                                                        prefix_expression.get_input(),
                                                        prefix_expression
                                                            .get_input()
                                                            .lines()
                                                            .take(
                                                                prefix_expression.line_col().0 - 1,
                                                            )
                                                            .map(|s| s.len() + 1)
                                                            .sum::<usize>()
                                                            + prefix_expression.line_col().1
                                                            - 1,
                                                    )
                                                    .unwrap(),
                                                )
                                                .with_path(
                                                    &self
                                                        .file_path
                                                        .borrow()
                                                        .clone()
                                                        .unwrap_or(String::new()),
                                                ),
                                            )
                                            .into());
                                        }
                                        Vec::new()
                                    },
                                }),
                                if let Some(ret_type) = ret_type {
                                    ret_type
                                } else {
                                    TypeAnnotation(vec![Type::Nil])
                                },
                            )
                        } else {
                            unreachable!()
                        }
                    } else if typ.0.iter().any(|f| matches!(f, Type::Any)) {
                        (
                            ExpressionNode::FunctionCall(FunctionCallNode {
                                identifier: Box::new(exp),
                                arguments: if let Some(args) = arguments {
                                    let args =
                                        args.into_inner()
                                            .map(|exp| self.build_expression(exp.into_inner()))
                                            .collect::<Result<
                                                Vec<(ExpressionNode, TypeAnnotation)>,
                                                ParseError,
                                            >>()?;
                                    args.into_iter().map(|f| f.0).collect()
                                } else {
                                    Vec::new()
                                },
                            }),
                            TypeAnnotation(vec![Type::Any]),
                        )
                    } else {
                        return Err(Box::new(
                            Error::new_from_pos(
                                ErrorVariant::CustomError {
                                    message: "Object is not callable".to_string(),
                                },
                                Position::new(
                                    prefix_expression.get_input(),
                                    prefix_expression
                                        .get_input()
                                        .lines()
                                        .take(prefix_expression.line_col().0 - 1)
                                        .map(|s| s.len() + 1)
                                        .sum::<usize>()
                                        + prefix_expression.line_col().1
                                        - 1,
                                )
                                .unwrap(),
                            )
                            .with_path(&self.file_path.borrow().clone().unwrap_or(String::new())),
                        )
                        .into());
                    })
                }
                Rule::expression | Rule::prefixExpression => {
                    self.build_expression(primary.into_inner())
                }
                Rule::Nil => Ok((
                    ExpressionNode::Literal(Object::Nil),
                    TypeAnnotation(vec![Type::Any]),
                )),
                Rule::Table => {
                    let mut hashmap = HashMap::new();
                    primary
                        .into_inner()
                        .try_for_each(|f| -> Result<(), ParseError> {
                            let mut inner = f.into_inner();
                            let literal_node = inner.next().unwrap().into_inner().next().unwrap();
                            let key = match literal_node.as_rule() {
                                Rule::Int => Literal::Int(literal_node.as_str().parse().unwrap()),
                                Rule::Float => {
                                    Literal::Float(literal_node.as_str().parse().unwrap())
                                }
                                Rule::Bool => Literal::Bool(literal_node.as_str().parse().unwrap()),
                                Rule::String => Literal::Str(
                                    literal_node.as_str().trim_matches('"').to_string(),
                                ),
                                _ => unreachable!(),
                            };
                            let value = self.build_expression(inner)?.0;
                            hashmap.insert(key, value);
                            Ok(())
                        })?;
                    Ok((
                        ExpressionNode::Literal(Object::Table(hashmap)),
                        TypeAnnotation(vec![Type::Table(vec![])]),
                    ))
                }
                Rule::array => {
                    let mut hashmap = HashMap::new();
                    for (i, value) in primary.into_inner().enumerate() {
                        let inner = value.into_inner();
                        let value = self.build_expression(inner)?;
                        hashmap.insert(Literal::Int(i as i64), value.0);
                    }
                    Ok((
                        ExpressionNode::Literal(Object::Table(hashmap)),
                        TypeAnnotation(vec![Type::Table(vec![])]),
                    ))
                }
                Rule::anonymousFunction => {
                    let mut inner_pairs = primary.into_inner();
                    let args = inner_pairs.next().unwrap();
                    let parameters = args
                        .into_inner()
                        .map(|pair| {
                            let mut inner = pair.into_inner();
                            Ok((
                                inner.next().unwrap().as_str().to_string(),
                                self.get_type(inner.next().unwrap())?,
                            ))
                        })
                        .collect::<Result<Vec<(String, Vec<Type>)>, ParseError>>()?;

                    let parameters: Vec<(String, TypeAnnotation)> = parameters
                        .into_iter()
                        .map(|(name, typ)| (name, TypeAnnotation(typ)))
                        .collect();

                    self.symbols.enter_scope();
                    parameters
                        .clone()
                        .into_iter()
                        .for_each(|(name, var_type)| self.symbols.add_variable(name, var_type));
                    let expression = inner_pairs.next().unwrap();
                    let (return_type, body) = if expression.as_rule() == Rule::functionReturn {
                        let ret_type = Some(TypeAnnotation(
                            self.get_type(expression.into_inner().next().unwrap())?,
                        ));
                        self.symbols
                            .current_fn_scope
                            .borrow_mut()
                            .push(ret_type.clone());
                        (ret_type, self.build_ast(inner_pairs.next().unwrap())?)
                    } else {
                        self.symbols.current_fn_scope.borrow_mut().push(None);
                        (None, self.build_ast(expression)?)
                    };

                    self.symbols.current_fn_scope.borrow_mut().pop();
                    self.symbols.exit_scope();
                    Ok((
                        ExpressionNode::AnonymousFunction(AnonymousFunctionNode {
                            return_type: return_type.clone(),
                            parameters: parameters.clone(),
                            body: Box::new(body),
                        }),
                        TypeAnnotation(vec![Type::Function(
                            parameters.into_iter().map(|f| f.1).collect(),
                            return_type,
                        )]),
                    ))
                }
                rule => unreachable!("expected Term, found: {rule:?}"),
            })
            .map_infix(|lhs, op, rhs| {
                let (rhs, rhs_type) = rhs?;
                let (lhs, lhs_type) = lhs?;

                let err = if matches!(op.as_rule(), Rule::logicalAnd | Rule::logicalOr) {
                    !rhs_type
                        .0
                        .iter()
                        .any(|f| matches!(f, Type::Any | Type::Table(_) | Type::Bool))
                        || !lhs_type
                            .0
                            .iter()
                            .any(|f| matches!(f, Type::Any | Type::Table(_) | Type::Bool))
                } else if match op.as_rule() {
                    Rule::ComparisonOperator => matches!(op.as_str(), "==" | "!="),
                    _ => false,
                } {
                    !(rhs_type
                        .0
                        .iter()
                        .any(|f| matches!(f, Type::Any | Type::Table(_)))
                        || lhs_type
                            .0
                            .iter()
                            .any(|f| matches!(f, Type::Any | Type::Table(_)))
                        || !rhs_type.find_duplicates(&lhs_type).0.is_empty()
                        || lhs_type.0.iter().any(|f| matches!(f, Type::Float))
                            && rhs_type.0.iter().any(|f| matches!(f, Type::Int))
                        || lhs_type.0.iter().any(|f| matches!(f, Type::Int))
                            && rhs_type.0.iter().any(|f| matches!(f, Type::Float)))
                } else {
                    let duplicates = rhs_type.find_duplicates(&lhs_type).0;

                    !(rhs_type
                        .0
                        .iter()
                        .any(|f| matches!(f, Type::Any | Type::Table(_)))
                        || lhs_type
                            .0
                            .iter()
                            .any(|f| matches!(f, Type::Any | Type::Table(_)))
                        || lhs_type.0.iter().any(|f| matches!(f, Type::Float))
                            && rhs_type.0.iter().any(|f| matches!(f, Type::Int))
                        || lhs_type.0.iter().any(|f| matches!(f, Type::Int))
                            && rhs_type.0.iter().any(|f| matches!(f, Type::Float))
                        || !duplicates.is_empty())
                };

                if err {
                    return Err(Box::new(
                        Error::new_from_pos(
                            ErrorVariant::CustomError {
                                message: format!("Cannot apply '{}' here", op.as_str()),
                            },
                            Position::new(
                                op.get_input(),
                                op.get_input()
                                    .lines()
                                    .take(op.line_col().0 - 1)
                                    .map(|s| s.len() + 1)
                                    .sum::<usize>()
                                    + op.line_col().1
                                    - 1,
                            )
                            .unwrap(),
                        )
                        .with_path(&self.file_path.borrow().clone().unwrap_or(String::new())),
                    )
                    .into());
                }

                let arithm_type = if lhs_type
                    .0
                    .iter()
                    .any(|f| matches!(f, Type::Any | Type::Table(_)))
                    || rhs_type
                        .0
                        .iter()
                        .any(|f| matches!(f, Type::Any | Type::Table(_)))
                {
                    TypeAnnotation(vec![Type::Any])
                } else if lhs_type.0.iter().any(|f| f == &Type::Str)
                    && rhs_type.0.iter().any(|f| f == &Type::Str)
                {
                    TypeAnnotation(vec![Type::Str])
                } else if lhs_type.0.iter().any(|f| f == &Type::Float)
                    || rhs_type.0.iter().any(|f| f == &Type::Float)
                {
                    TypeAnnotation(vec![Type::Float])
                } else {
                    TypeAnnotation(vec![Type::Int])
                };

                let logical_type = if lhs_type
                    .0
                    .iter()
                    .any(|f| matches!(f, Type::Any | Type::Table(_)))
                    || rhs_type
                        .0
                        .iter()
                        .any(|f| matches!(f, Type::Any | Type::Table(_)))
                {
                    TypeAnnotation(vec![Type::Any])
                } else {
                    TypeAnnotation(vec![Type::Bool])
                };

                let (op, res_type) = match op.as_rule() {
                    Rule::ComparisonOperator => (
                        match op.as_str() {
                            "==" => BinaryOperator::Equal,
                            "!=" => BinaryOperator::NotEqual,
                            ">" => BinaryOperator::GreaterThan,
                            ">=" => BinaryOperator::GreaterThanOrEqual,
                            "<" => BinaryOperator::LessThan,
                            "<=" => BinaryOperator::LessThanOrEqual,
                            str => unreachable!("expected Operation, found: {str}"),
                        },
                        logical_type,
                    ),
                    Rule::logicalAnd => (BinaryOperator::LogicalAnd, logical_type),
                    Rule::logicalOr => (BinaryOperator::LogicalOr, logical_type),
                    Rule::Add => (BinaryOperator::Add, arithm_type),
                    Rule::Subtract => (BinaryOperator::Substract, arithm_type),
                    Rule::Multiply => (BinaryOperator::Multiply, arithm_type),
                    Rule::Divide => (BinaryOperator::Divide, arithm_type),
                    Rule::Power => (BinaryOperator::Power, arithm_type),
                    Rule::Mod => (
                        BinaryOperator::Mod,
                        if lhs_type
                            .0
                            .iter()
                            .any(|f| matches!(f, Type::Any | Type::Table(_)))
                            || rhs_type
                                .0
                                .iter()
                                .any(|f| matches!(f, Type::Any | Type::Table(_)))
                        {
                            TypeAnnotation(vec![Type::Any])
                        } else {
                            TypeAnnotation(vec![Type::Int])
                        },
                    ),
                    Rule::Div => (
                        BinaryOperator::Div,
                        if lhs_type
                            .0
                            .iter()
                            .any(|f| matches!(f, Type::Any | Type::Table(_)))
                            || rhs_type
                                .0
                                .iter()
                                .any(|f| matches!(f, Type::Any | Type::Table(_)))
                        {
                            TypeAnnotation(vec![Type::Any])
                        } else {
                            TypeAnnotation(vec![Type::Int])
                        },
                    ),
                    rule => unreachable!("expected Operation, found: {rule:?}"),
                };
                Ok((
                    ExpressionNode::BinaryOperation(Box::new(lhs), op, Box::new(rhs)),
                    res_type,
                ))
            })
            .map_prefix(|op, rhs| match op.as_rule() {
                Rule::logicalNot => {
                    let (rhs, typ) = rhs?;
                    if !typ
                        .0
                        .iter()
                        .any(|f| matches!(f, Type::Any | Type::Table(_) | Type::Bool))
                    {
                        return Err(Box::new(
                            Error::new_from_pos(
                                ErrorVariant::CustomError {
                                    message: "Cannot apply '!' here".to_string(),
                                },
                                Position::new(
                                    op.get_input(),
                                    op.get_input()
                                        .lines()
                                        .take(op.line_col().0 - 1)
                                        .map(|s| s.len() + 1)
                                        .sum::<usize>()
                                        + op.line_col().1
                                        - 1,
                                )
                                .unwrap(),
                            )
                            .with_path(&self.file_path.borrow().clone().unwrap_or(String::new())),
                        )
                        .into());
                    }
                    Ok((
                        ExpressionNode::UnaryOperation(UnaryOperator::LogicalNot, Box::new(rhs)),
                        typ,
                    ))
                }
                Rule::unaryMinus => {
                    let (rhs, typ) = rhs?;
                    if !typ
                        .0
                        .iter()
                        .any(|f| matches!(f, Type::Any | Type::Table(_) | Type::Float | Type::Int))
                    {
                        return Err(Box::new(
                            Error::new_from_pos(
                                ErrorVariant::CustomError {
                                    message: "Cannot apply '-' here".to_string(),
                                },
                                Position::new(
                                    op.get_input(),
                                    op.get_input()
                                        .lines()
                                        .take(op.line_col().0 - 1)
                                        .map(|s| s.len() + 1)
                                        .sum::<usize>()
                                        + op.line_col().1
                                        - 1,
                                )
                                .unwrap(),
                            )
                            .with_path(&self.file_path.borrow().clone().unwrap_or(String::new())),
                        )
                        .into());
                    }
                    Ok((
                        ExpressionNode::UnaryOperation(UnaryOperator::Negate, Box::new(rhs)),
                        typ,
                    ))
                }
                _ => unreachable!(),
            })
            .map_postfix(|lhs, op| match op.as_rule() {
                Rule::index => {
                    let (lhs, typ) = lhs?;
                    if !typ
                        .0
                        .iter()
                        .any(|f| matches!(f, Type::Any | Type::Table(_)))
                    {
                        return Err(Box::new(
                            Error::new_from_pos(
                                ErrorVariant::CustomError {
                                    message: "Index can only be applied to tables".to_string(),
                                },
                                Position::new(
                                    op.get_input(),
                                    op.get_input()
                                        .lines()
                                        .take(op.line_col().0 - 1)
                                        .map(|s| s.len() + 1)
                                        .sum::<usize>()
                                        + op.line_col().1
                                        - 1,
                                )
                                .unwrap(),
                            )
                            .with_path(&self.file_path.borrow().clone().unwrap_or(String::new())),
                        )
                        .into());
                    }
                    Ok((
                        ExpressionNode::UnaryOperation(
                            UnaryOperator::Index(Box::new(
                                self.build_expression(op.into_inner())?.0,
                            )),
                            Box::new(lhs),
                        ),
                        TypeAnnotation(vec![Type::Any]),
                    ))
                }
                _ => unreachable!(),
            })
            .parse(pairs)
    }
}
