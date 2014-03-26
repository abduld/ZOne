
part of zone.backend;

class JavascriptFunctionInstructionVisitor implements InstructionVisitor {
  @override
  String visitLambdaInstruction(LambdaInstruction inst) {
    // TODO: implement visitLambdaInstruction
    return "${inst.target} = function(${inst.args.map((arg) => arg.toString()).join(", ")}) {\n ${ToJavaScriptCode(inst.body)} \n}\n";
  }

  @override
  String visitBinaryInstruction(BinaryInstruction inst) {
    return "";
  }

  @override
  Object visitBranchInstruction(BranchInstruction inst) {
    return "";
  }

  @override
  Object visitCallInstruction(CallInstruction inst) {
    return "";
  }

  @override
  Object visitEmptyInstruction(EmptyInstruction inst) {
    return "";
  }

  @override
  Object visitMapInstruction(MapInstruction inst) {
    return "";
  }

  @override
  Object visitOpCode(OpCode op) {
    return "";
  }

  @override
  Object visitReduceInstruction(ReduceInstruction inst) {
    return "";
  }

  @override
  Object visitReturnInstruction(ReturnInstruction inst) {
    return "";
  }

  @override
  Object visitUnaryInstruction(UnaryInstruction inst) {
    return "";
  }

  @override
  Object visitValue(Value val) {
    return "";
  }
}

class JavascriptInstructionVisitor implements InstructionVisitor {
  @override
  String visitBinaryInstruction(BinaryInstruction inst) {
    // TODO: implement visitBinaryInstruction
    return '${inst.op.accept(this)}(${inst.args.map((arg) => arg.accept(this)).join(", ")})';
  }

  @override
  String visitBranchInstruction(BranchInstruction inst) {
    return '${inst.op.accept(this)}(${inst.args.map((arg) => arg.accept(this)).join(", ")})';
  }

  final Map<String, String> infixFunctions = {
    "Plus" : "+"
  };
  
  String infixSymbol(SymbolValue s) =>
      infixFunctions[s.value];
  
  bool infixFunctionQ(SymbolValue s) =>
      infixFunctions.containsKey(s.value);
  
  @override
  String visitCallInstruction(CallInstruction inst) {
    if (SystemSymbolQ(inst.function) && infixFunctionQ(inst.function)) {
      return '${inst.target} = ${inst.fargs.map((arg) => arg.accept(this)).join(infixSymbol(inst.function))}';
    }
    return '${inst.target} = ${inst.function.accept(this)}(${inst.fargs.map((arg) => arg.accept(this)).join(", ")})';
  }

  @override
  String visitEmptyInstruction(EmptyInstruction inst) {
    // TODO: implement visitEmptyInstruction
    return "todoEmpty";
  }

  @override
  String visitLambdaInstruction(LambdaInstruction inst) {
    // TODO: implement visitLambdaInstruction
    return "todoLambda";
  }

  @override
  String visitMapInstruction(MapInstruction inst) {
    // TODO: implement visitMapInstruction
    return "todoMap";
  }

  @override
  String visitOpCode(OpCode op) {
    return op.name;
  }

  @override
  String visitReduceInstruction(ReduceInstruction inst) {
    // TODO: implement visitReduceInstruction
    return "todoReduce";
  }

  @override
  String visitReturnInstruction(ReturnInstruction inst) {
    // TODO: implement visitReturnInstruction
    return "todoReturn";
  }

  @override
  String visitUnaryInstruction(UnaryInstruction inst) {
    // TODO: implement visitUnaryInstruction
    if (inst.op == LoadOp) {
      return "${inst.target} = ${inst.args[0]}";
    }
    return "todoUnary  :: " + inst.toString();
  }

  @override
  String visitValue(Value val) {
    if (val is SymbolValue) {
       if (SystemSymbolQ(val)) {
         return val.toString() + "Symbo";
       }
    } else if (val is IntegerValue || val is RealValue) {
      return val.toString();
    }
    return "todoValue";
  }
}

String ToJavaScriptCode(List<Instruction> insts) {
  JavascriptFunctionInstructionVisitor funcVisitor = new JavascriptFunctionInstructionVisitor();
  JavascriptInstructionVisitor visitor = new JavascriptInstructionVisitor();
  String res = insts.map((inst) => inst.accept(funcVisitor)).join("") +
      insts.map((inst) => inst.accept(visitor)).join(";\n") + ";";
  return res;
}

