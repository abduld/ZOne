
part of zone.analysis;


class DefVisitor extends InstructionVisitor {
  Map<Instruction, Value> defs = {};

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
    DefVisitor vst = new DefVisitor();
    List<Instruction> res = insts.map((nd) => nd.accept(vst)).toList(growable: false);
    defs = vst.defs;
    return res;
  }
}

class UsePass implements InstructionPass {
  Map<Instruction, List<Value>> uses;
  List<Instruction> run(List<Instruction> insts) {
    UseVisitor vst = new UseVisitor();
    List<Instruction> res = insts.map((nd) => nd.accept(vst)).toList(growable: false);
    uses = vst.uses;
    return res;
  }
}
