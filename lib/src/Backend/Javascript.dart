
part of zone.backend;

class JavascriptFunctionInstructionVisitor extends InstructionVisitor {
  @override
  String visitLambdaInstruction(LambdaInstruction inst) {
    // TODO: implement visitLambdaInstruction
    String res = "var ${inst.target} = function(${inst.args.map((arg) => arg.toString()).join(", ")}) {\n" +
        "${iToJavaScriptCode(inst.body)}" +
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
  
  String processMap(CallInstruction inst) {
    String f = inst.fargs[0].accept(this);
    String args = inst.fargs.skip(1).map((arg) => arg.accept(this)).join(", ");
    return "var ${inst.target} = _.map(${f}, ${args})";
  }

  String processReduce(CallInstruction inst) {
    return "todoReduce";
  }
  
  @override
  String visitCallInstruction(CallInstruction inst) {
    if (SystemSymbolQ(inst.function)) {
      if (isReturn(inst.function)) {
        return "return(${inst.fargs.map((arg) => arg.accept(this)).join(", ")})";
      } else if (isMap(inst.function)) {
        return processMap(inst);
      } else if (isReduce(inst.function)) {
        return processReduce(inst);
      } else if (infixFunctionQ(inst.function)) {
        return 'var ${inst.target} = ${inst.fargs.map((arg) => arg.accept(this)).join(infixSymbol(inst.function))}';
      }
    }
    return 'var ${inst.target} = ${inst.function.accept(this)}(${inst.fargs.map((arg) => arg.accept(this)).join(", ")})';
  }

  @override
  String visitEmptyInstruction(EmptyInstruction inst) {
    return "";
  }

  @override
  String visitLambdaInstruction(LambdaInstruction inst) {
    return "";
  }

  @override
  String visitOpCode(OpCode op) {
    return op.name;
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
      return val.toString();
    }
    print("XXX " + val.toString());
    return "todoValue";
  }
}

String iToJavaScriptCode(List<Instruction> insts) {
  JavascriptFunctionInstructionVisitor funcVisitor = new JavascriptFunctionInstructionVisitor();
  JavascriptInstructionVisitor visitor = new JavascriptInstructionVisitor();
  String res = insts.map((inst) => inst.accept(funcVisitor)).join("") +
      insts.map((inst) => inst.accept(visitor)).where((String line) => line.trim() != "").join(";\n") + ";";
  return res;
}

String ToJavaScriptCode(List<Instruction> insts) {
  return "var _ = require('underscore');\n" + iToJavaScriptCode(insts);
}

