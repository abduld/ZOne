
part of zone.analysis;

class LiftFunctionVisitor extends InstructionVisitor {
  Map<Value, List<Value>> extaArgs = {};
  Map<LambdaInstruction, List<Value>> freeVariables;

  // Free variable map probably comes from FreeVariables analysis pass.
  LiftFunctionVisitor(insts, Map<LambdaInstruction, List<Value>> freeVariables) 
    : this.freeVariables = freeVariables, super(insts) {
  }
  
  @override
  Instruction visitCallInstruction(CallInstruction inst) {
    if (extaArgs.containsKey(inst.function)) {
      inst.args.addAll(extaArgs[inst.function]);
      print("LiftFunctionVisitor.visitCallInstruction: added args to $inst");
    }
    return inst;
  }

 
  @override
  Instruction visitLambdaInstruction(LambdaInstruction inst) {
    print("LiftFunctionVisitor.visitLambdaInstruction: $inst");
    print("LiftFunctionVisitor.visitLambdaInstruction: $freeVariables");
    List<Value> freeVars = freeVariables[inst];
    inst.args.addAll(freeVars);
    return inst; 
  }
  
  @override
  Object visitDefault(Object obj) => obj;

}

// Running lift function pass automatically does a free variable analysis pass.
class LiftFunctionPass implements InstructionPass {
  List<Instruction> run(List<Instruction> insts) {
    FreeVariablesPass fvp = new FreeVariablesPass();
    fvp.run(insts);
    //print(fvp.free);
    LiftFunctionVisitor vst = new LiftFunctionVisitor(insts, fvp.free);
    return vst.out;
  }
}


