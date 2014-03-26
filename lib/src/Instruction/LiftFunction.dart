part of zone.instruction;


class LiftFunctionVisitor implements InstructionVisitor {
  Map<Value, List<Value>> extaArgs = {};

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
  Instruction visitBinaryInstruction(BinaryInstruction inst) => inst;

  @override
  Instruction visitBranchInstruction(BranchInstruction inst) => inst;

  @override
  Instruction visitEmptyInstruction(EmptyInstruction inst) => inst;

  @override
  Instruction visitMapInstruction(MapInstruction inst) => inst;

  @override
  OpCode visitOpCode(OpCode op) => op;

  @override
  Instruction visitReduceInstruction(ReduceInstruction inst) => inst;

  @override
  Instruction visitReturnInstruction(ReturnInstruction inst) => inst;

  @override
  Instruction visitUnaryInstruction(UnaryInstruction inst) => inst;

  @override
  Value visitValue(Value val) => val;
}


List<Instruction> LiftFunctionPass(List<Instruction> insts) {
  LiftFunctionVisitor vst = new LiftFunctionVisitor();
  List<Instruction> res = insts.map((nd) => nd.accept(vst)).toList(growable: false);
  return res;
}