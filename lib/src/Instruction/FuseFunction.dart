part of zone.instruction;


class FuseMapVisitor extends InstructionVisitor {
  Map<Value, List<Value>> extaArgs = {};
  Map<Instruction, List<Value>> uses = {};

  FuseMapVisitor(insts) : super(insts);

  @override
  void depends() {
    UseVisitor vst = new UseVisitor(instructions);
    uses = vst.uses;
  }

  @override
  Instruction visitCallInstruction(CallInstruction inst) {
    if (isMap(inst.function)) {
      Value target = inst.target;
      
    }
    return inst;
  }
  
  @override
  Instruction visitLambdaInstruction(LambdaInstruction inst) {
    return inst; 
  }
  
  @override
  Object visitDefault(Object obj) {
    return obj;
  }

}