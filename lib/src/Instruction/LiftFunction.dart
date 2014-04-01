part of zone.instruction;


class LiftFunctionVisitor extends InstructionVisitor {
  Map<Value, List<Value>> extaArgs = {};

  LiftFunctionVisitor(insts) : super(insts);
  
  @override
  Instruction visitCallInstruction(CallInstruction inst) {
    if (extaArgs.containsKey(inst.function)) {
      inst.args.addAll(extaArgs[inst.function]);
    }
    return inst;
  }

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
  
  @override
  Instruction visitLambdaInstruction(LambdaInstruction inst) {
    List<Value> freeVars = freeVariables(inst);
    inst.args.addAll(freeVars);
    extaArgs[inst.target] = freeVars;
    return inst; 
  }
  
  @override
  Object visitDefault(Object obj) => obj;

}


class LiftFunctionPass implements InstructionPass {
  List<Instruction> run(List<Instruction> insts) {
    LiftFunctionVisitor vst = new LiftFunctionVisitor(insts);
    return vst.out;
  }
}
