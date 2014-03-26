
part of zone.backend;

class JavascriptFunctionInstructionVisitor extends InstructionVisitor {
  @override
  String visitLambdaInstruction(LambdaInstruction inst) {
    // TODO: implement visitLambdaInstruction
    String res = "${inst.target} = function(${inst.args.map((arg) => arg.toString()).join(", ")}) {\n" +
        "${ToJavaScriptCode(inst.body)}" +
        "\n}\n";
     return res;
  }

  @override
  String visitDefault(Object obj) => "";
}

class JavascriptInstructionVisitor extends InstructionVisitor {
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
      return 'var ${inst.target} = ${inst.fargs.map((arg) => arg.accept(this)).join(infixSymbol(inst.function))}';
    } else if (SystemSymbolQ(inst.function) && isReturn(inst.function)){
      return "return(${inst.fargs.map((arg) => arg.accept(this)).join(", ")})";
    }
    return 'var ${inst.target} = ${inst.function.accept(this)}(${inst.fargs.map((arg) => arg.accept(this)).join(", ")})';
  }

  @override
  String visitEmptyInstruction(EmptyInstruction inst) {
    // TODO: implement visitEmptyInstruction
    return "todoEmpty";
  }

  @override
  String visitLambdaInstruction(LambdaInstruction inst) {
    return "";
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
      return "var ${inst.target} = ${inst.args[0]}";
    }
    return "todoUnary  :: " + inst.toString();
  }

  @override
  String visitValue(Value val) {
    if (val is IdentifierValue) {
      return val.toString();
    } if (val is SymbolValue) {
       if (SystemSymbolQ(val)) {
         if (val.value == "Return") {
           return "return";
         }
         return val.toString() + "Symbo";
       }
    } else if (val is IntegerValue || val is RealValue) {
      return val.toString();
    } else if (val is ListValue) {
      print("XXX List " + val.toString());
      return val.toString();
    }
    print("XXX " + val.toString());
    return "todoValue";
  }
}

String ToJavaScriptCode(List<Instruction> insts) {
  JavascriptFunctionInstructionVisitor funcVisitor = new JavascriptFunctionInstructionVisitor();
  JavascriptInstructionVisitor visitor = new JavascriptInstructionVisitor();
  String res = insts.map((inst) => inst.accept(funcVisitor)).join("") +
      insts.map((inst) => inst.accept(visitor)).where((String line) => line.trim() != "").join(";\n") + ";";
  return res;
}

