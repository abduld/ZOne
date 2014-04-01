
part of zone.analysis;


class DefVisitor extends InstructionVisitor {
  Map<Instruction, Value> defs = {};

  DefVisitor(insts) : super(insts);
  
  @override
  Object visitDefault(Object obj) {
    if (obj is Instruction && obj.target != null) {
      defs[obj] = obj.target;
    }
    return obj;
  }

}

class UseVisitor extends InstructionVisitor {
  Map<Instruction, List<Value>> uses = {};

  UseVisitor(insts) : super(insts);
  
  @override
  Object visitDefault(Object obj) {
    if (obj is Instruction && obj.args != null) {
      if (obj is CallInstruction) {
        uses[obj] = obj.fargs;
      } else {
        uses[obj] = obj.args;
      }
    }
    return obj;
  }

}

class DefPass implements InstructionPass {
  Map<Instruction, Value> defs;
  List<Instruction> run(List<Instruction> insts) {
    DefVisitor vst = new DefVisitor(insts);
    defs = vst.defs;
    return vst.out;
  }
}

class UsePass implements InstructionPass {
  Map<Instruction, List<Value>> uses;
  List<Instruction> run(List<Instruction> insts) {
    UseVisitor vst = new UseVisitor(insts);
    uses = vst.uses;
    return vst.out;
  }
}
