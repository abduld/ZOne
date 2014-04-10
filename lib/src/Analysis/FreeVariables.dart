
part of zone.analysis;

class FreeVariablesVisitor extends InstructionVisitor {
  Map<LambdaInstruction, List<Value>> free = {};

  FreeVariablesVisitor(insts) : super(insts);
  
  @override
  Instruction visitLambdaInstruction(LambdaInstruction inst) {
    List<Value> freeVars = freeVariables(inst);
    inst.args.addAll(freeVars);
    free[inst] = freeVars;
    return inst; 
  }
  
  @override
  Object visitDefault(Object obj) => obj;
  
  List<Value> freeVariables(LambdaInstruction inst) {
    List<Value> bound = new List<Value>.from(inst.args, growable: true);
    List<Value> free = [];
    inst.body.forEach((Instruction line) {
      line.args.forEach((arg) {
        if (arg is IdentifierValue && !bound.any((b) => arg.sameQ(b))) {
          free.add(arg);
        }
      });
      bound.add(line.target);
    });
    return free;
  }
}


class FreeVariablesPass implements InstructionPass {
  Map<LambdaInstruction, List<Value>> free;
  List<Instruction> run(List<Instruction> insts) {
    FreeVariablesVisitor vst = new FreeVariablesVisitor(insts);
    free  = vst.free;
    return vst.out;
  }
}


