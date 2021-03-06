part of zone.instruction;


class PeepholeOptimizeVisitor extends InstructionVisitor {
  Object prev = null;
  Map<String, IdentifierValue> rename = {};
  
  PeepholeOptimizeVisitor(insts) : super(insts);
  
  @override
  Instruction visitUnaryInstruction(UnaryInstruction inst) {
    if (inst.op == LoadOp) {
      /* Catch the patter 
       *     x = ...;
       *     g = x;
       * and replace all occurences of g with x
       */
      if (prev is UnaryInstruction || prev is CallInstruction) {
        Instruction prevInst = prev as Instruction;
        if (prevInst.target.sameQ(inst.args[0])) {
          rename[inst.target.value] = new IdentifierValue(prevInst.target.value);
        }
      }
    }
    prev = inst;
    return null;
  }

  
  @override
  Object visitValue(Value val) {
    if (val is IdentifierValue && rename.containsKey(val)) {
      return rename[val];
    }
    return val;
  }
  
  @override
  Object visitDefault(Object obj) {
    prev = obj;
    if (obj is Instruction) {
      obj.args = obj.args.map((arg) {
        if (arg is IdentifierValue && rename.containsKey(arg.value)) {
          return rename[arg.value];
        } else {
          return arg;
        }
      }).toList();
    }
    return obj;
  }
}

class PeepholeOptimizePass implements InstructionPass {
  List<Instruction> run(List<Instruction> insts) {
    PeepholeOptimizeVisitor vst = new PeepholeOptimizeVisitor(insts);
    return vst.out;
  }
}