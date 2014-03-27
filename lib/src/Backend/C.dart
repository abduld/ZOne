
part of zone.backend;

class CInstructionVisitor extends InstructionVisitor {
  @override
  String visitBinaryInstruction(BinaryInstruction inst) {
    // TODO: implement visitBinaryInstruction
    return "todo";
  }

  @override
  String visitBranchInstruction(BranchInstruction inst) {
    // TODO: implement visitBranchInstruction
    return "todo";
  }

  @override
  String visitCallInstruction(CallInstruction inst) {
    // TODO: implement visitCallInstruction
    return "todo";
  }

  @override
  String visitEmptyInstruction(EmptyInstruction inst) {
    // TODO: implement visitEmptyInstruction
    return "todo";
  }

  @override
  String visitLambdaInstruction(LambdaInstruction inst) {
    // TODO: implement visitLambdaInstruction
    return "todo";
  }

  @override
  String visitOpCode(OpCode op) {
    // TODO: implement visitOpCode
    return "todo";
  }

  @override
  String visitUnaryInstruction(UnaryInstruction inst) {
    // TODO: implement visitUnaryInstruction
    return "todo";
  }

  @override
  String visitValue(Value val) {
    // TODO: implement visitValue
    return "todo";
  }
}

String ToCCode(List<Instruction> insts) {
  return "todo";
}

