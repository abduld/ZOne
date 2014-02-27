
library zone.lower;

import '../AST/AST.dart';
import '../Instruction/Instruction.dart';



class _Lower implements ASTNodeVisitor {
  List<Instruction> insts;
  Map<String, int> counters = {};
  
  _Lower() {
    insts = [];
  }
  
  String genSym(String x) {
    int n = counters.putIfAbsent(x, () => 0);
    counters[x] = n+1;
    return "$x$n";
  }

  int get hashCode => insts.hashCode;
  bool sameQ(other) => hashCode == other.hashCode;
  
  List<Instruction> get instructions => insts;
  
  Object visitAssignmentNode(AssignmentNode ass) {
    print("TODO :: " + ass.toString());
    return null;
    // TODO: implement visitAssignmentNode
  }

  Object visitAtomNode(AtomNode atm) {
    if (atm is IntegerNode) {
      return new IntegerValue(atm.value);
    } else if (atm is RealNode) {
      return new RealValue(atm.value);
    } else if (atm is TrueNode) {
      return new TrueValue();
    } else if (atm is FalseNode) {
      return new FalseValue();
    } else if (atm is SymbolNode) {
      return new SymbolValue(atm.value);
    } else if (atm is StringNode) {
      return new StringValue(atm.value);
    } else if (atm is IdentifierNode) {
      return new StringValue(atm.value);
    } else if (atm is ListNode) {
      List<Value> lst = atm.value.map(
          (elem) => elem.accept(this)
      ).toList();
      return new ListValue(lst);
    }
    return new UnknownValue(atm.toString());
  }

  Object visitCallNode(CallNode call) {
    Instruction inst;
    String name = call.name;
    IdentifierValue f = new IdentifierValue(name);
    IdentifierValue lhs = new IdentifierValue(genSym(name));
    
    List<Value> args = call.args.map((arg) => arg.accept(this)).toList();
    inst = new CallInstruction(lhs, f, args);
    
    emit(inst);

    return lhs;
  }

  Object visitDocumentationNode(DocumentationNode doc) {
    print("TODO :: " + doc.toString());
    return null;
    // TODO: implement visitDocumentationNode
  }

  Object visitFunctionDeclarationNode(FunctionDeclarationNode fundecl) {
    Instruction inst;
    List<Instruction> body;
    IdentifierValue lhs = new IdentifierValue(genSym("lambda"));
    _Lower vst = new _Lower();

    List<Value> args = fundecl.args.map((arg) => arg.accept(this)).toList();
    fundecl.visitChildren(vst);
    body = vst.instructions;
    body.add(new ReturnInstruction(body.last.target));;
    
    inst = new LambdaInstruction(lhs, args, body);
    
    emit(inst);
    
    return lhs;
  }

  Object visitParameterNode(ParameterNode param) {
    return new IdentifierValue(param.id.value, new TypeValue.fromTypeNode(param.type));
  }

  Object visitProgramNode(ProgramNode prog) {
    prog.visitChildren(this);
    return null;
  }

  Object visitTypeNode(TypeNode typ) {
    return new TypeValue.fromTypeNode(typ);
  }

  Object visitVariableDeclarationNode(VariableDeclarationNode decl) {
    Instruction inst;
    IdentifierValue lhs = new IdentifierValue(decl.lhs.value,
        new TypeValue.fromTypeNode(decl.type));
    
    if (decl.hasRHS) {
      Object rhs = decl.rhs.accept(this);
      inst = new UnaryInstruction(LoadOp, lhs, rhs);
    } else {
      inst = new EmptyInstruction(AllocaOp, lhs);
    }
    emit(inst);
    return lhs.value;
  }
  
  void emit(Instruction inst) {
    instructions.add(inst);
  }
}

List<Instruction> Lower(ASTNode nd) {
  _Lower vst = new _Lower();
  nd.accept(vst);
  print(vst.instructions.join('\n'));
  return vst.instructions;
}