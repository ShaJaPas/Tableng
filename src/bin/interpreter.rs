use std::{fs::File, io::Read};

use clap::Parser;
use tableng::{ir_gen, reg_alloc, vm_v2, CompileRegister};

#[derive(Parser)]
/// Tableng interpreter
struct Cli {
    /// path to file with Tableng source code
    #[arg(long, group = "input", short = 'p')]
    file_path: Option<String>,

    /// Tableng source code
    #[arg(long, group = "input", short = 'd')]
    data: Option<String>,
}

fn main() {
    let args = Cli::parse();

    let source = if let Some(ref path) = args.file_path {
        let file = File::open(path);
        match file {
            Ok(mut f) => {
                let mut source = String::new();
                f.read_to_string(&mut source).unwrap();
                source
            }
            Err(e) => {
                println!("Cannot read a file: {e}");
                return;
            }
        }
    } else if let Some(source) = args.data {
        source
    } else {
        println!("No data received, use one of the input options!");
        return;
    };

    let env = reg_alloc::get_env();
    match ir_gen::BytecodeBuilder::from_source(
        &source,
        args.file_path,
        env.preferred_regs_by_class[0].iter().max().unwrap().index(),
    ) {
        Ok(builder) => {
            if builder.instructions.is_empty() {
                return;
            }
            let allocs = regalloc2::run(&builder, &env, &reg_alloc::get_options()).unwrap();
            let code = builder.build(
                allocs
                    .allocs
                    .into_iter()
                    .map(|f| f.as_reg().unwrap())
                    .collect(),
            );
            let mut vm = vm_v2::VM::default();
            vm.run(code).map_err(|err| println!("{err}")).ok();
        }
        Err(err) => println!("{err}"),
    };
}
