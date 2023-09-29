use regalloc2::{
    Block, Function, Inst, InstRange, MachineEnv, Operand, PReg, PRegSet, RegClass,
    RegallocOptions, VReg,
};

use crate::{ir::IROpCode, ir_gen::BytecodeBuilder};

pub fn get_env() -> MachineEnv {
    MachineEnv {
        preferred_regs_by_class: [
            (0..32).map(|i| PReg::new(i, RegClass::Int)).collect(),
            vec![],
            vec![],
        ],
        non_preferred_regs_by_class: [vec![], vec![], vec![]],
        scratch_by_class: [None, None, None],
        fixed_stack_slots: vec![],
    }
}

pub fn get_options() -> RegallocOptions {
    RegallocOptions::default()
}

impl Function for BytecodeBuilder {
    fn num_insts(&self) -> usize {
        self.instructions.len()
    }

    fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    fn entry_block(&self) -> Block {
        Block(0)
    }

    fn block_insns(&self, block: Block) -> InstRange {
        InstRange::forward(
            Inst::new(self.blocks[block.index()].instructions.start),
            Inst::new(self.blocks[block.index()].instructions.end),
        )
    }

    fn block_succs(&self, block: Block) -> &[Block] {
        &self.blocks[block.index()].successors
    }

    fn block_preds(&self, block: Block) -> &[Block] {
        &self.blocks[block.index()].predecessors
    }

    fn block_params(&self, block: Block) -> &[VReg] {
        &self.blocks[block.index()].params
    }

    fn is_ret(&self, insn: Inst) -> bool {
        let inst = self.instructions[insn.index()];

        matches!(inst, IROpCode::ReturnNil | IROpCode::Return { .. })
    }

    fn is_branch(&self, insn: Inst) -> bool {
        let inst = self.instructions[insn.index()];
        matches!(inst, IROpCode::Jmp { .. })
    }

    fn branch_blockparams(&self, _block: Block, _insn: Inst, _succ_idx: usize) -> &[VReg] {
        &[]
    }

    fn inst_operands(&self, insn: Inst) -> &[Operand] {
        &self.operands[insn.index()]
    }

    fn inst_clobbers(&self, _insn: Inst) -> PRegSet {
        PRegSet::empty()
    }

    fn num_vregs(&self) -> usize {
        self.reg_num
    }

    fn spillslot_size(&self, regclass: RegClass) -> usize {
        match regclass {
            RegClass::Int => 1,
            RegClass::Float => 1,
            RegClass::Vector => 0,
        }
    }
}
