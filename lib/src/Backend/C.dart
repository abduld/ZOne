
part of zone.backend;

Map<String, String> typeMap = {
  "Void": "void",
  "Integer": "int",
  "Real": "float"
};

String toCType(TypeValue t) => typeMap[t.toString()];
String cType(Value v) => v == null ? "void" : toCType(v.type);

class CFunctionInstructionVisitor extends InstructionVisitor {
  @override
  String visitLambdaInstruction(LambdaInstruction inst) {
    String returnType = cType(inst.target);
    Value funName = inst.target;
    String args = inst.args.map((arg) => cType(arg) + " " + arg.toString()).join(", ");
    String body = iToCCode(inst.body);
    String res = "${returnType} ${funName}(${args}) {\n${body} \n}\n";
     return res;
  }

  @override
  String visitDefault(Object obj) => "";
}

class CInstructionVisitor extends InstructionVisitor {
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
    String returnType = cType(inst.fargs[0]);
    String args = inst.fargs.skip(1).map((arg) => arg.accept(this)).join(", ");
    return "${returnType} * ${inst.target} = doMap(${f}, ${args})";
  }

  String processReduce(CallInstruction inst) {
    return "todoReduce";
  }
  
  @override
  String visitCallInstruction(CallInstruction inst) {
    String varType = cType(inst.target);
    if (SystemSymbolQ(inst.function)) {
      if (isReturn(inst.function)) {
        return "return(${inst.fargs.map((arg) => arg.accept(this)).join(", ")})";
      } else if (isMap(inst.function)) {
        return processMap(inst);
      } else if (isReduce(inst.function)) {
        return processReduce(inst);
      } else if (infixFunctionQ(inst.function)) {
        return '${varType} ${inst.target} = ${inst.fargs.map((arg) => arg.accept(this)).join(infixSymbol(inst.function))}';
      }
    }
    return '${varType} ${inst.target} = ${inst.function.accept(this)}(${inst.fargs.map((arg) => arg.accept(this)).join(", ")})';
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
      String varType = cType(inst.target);
      return "${varType} ${inst.target} = ${inst.args[0]}";
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

String iToCCode(List<Instruction> insts) {
  CFunctionInstructionVisitor funcVisitor = new CFunctionInstructionVisitor();
  CInstructionVisitor visitor = new CInstructionVisitor();
  String res = insts.map((inst) => inst.accept(funcVisitor)).join("") +
      insts.map((inst) => inst.accept(visitor)).where((String line) => line.trim() != "").join(";\n") + ";";
  return res;
}

String ToCCode(List<Instruction> insts) {
  return "#include <stdio.h>\n" + iToCCode(insts);
}


