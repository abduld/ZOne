
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
    print(ass);
    return null;
    // TODO: implement visitAssignmentNode
  }

  Object visitAtomNode(AtomNode atm) {
    print(atm);
    return null;
    // TODO: implement visitAtomNode
  }

  Object visitCallNode(CallNode call) {
    Instruction inst;
    String name = call.name;
    IdentifierValue f = new IdentifierValue(name);
    IdentifierValue lhs = new IdentifierValue(genSym(name));
    
    List<Value> args;
    call.args.forEach((arg) => args.add(arg.accept(this)));
    inst = new CallInstruction(f, args);
    
    emit(inst);
    
    return lhs;
  }

  Object visitDocumentationNode(DocumentationNode doc) {
    print(doc);
    return null;
    // TODO: implement visitDocumentationNode
  }

  Object visitFunctionDeclarationNode(FunctionDeclarationNode fundecl) {
    print(fundecl);
    return null;
    // TODO: implement visitFunctionDeclarationNode
  }

  Object visitParameterNode(ParameterNode param) {
    print(param);
    return null;
    // TODO: implement visitParameterNode
  }

  Object visitProgramNode(ProgramNode prog) {
    print(prog);
    prog.visitChildren(this);
    return null;
    // TODO: implement visitProgramNode
  }

  Object visitTypeNode(TypeNode typ) {
    print(typ);
    return null;
    // TODO: implement visitTypeNode
  }

  Object visitVariableDeclarationNode(VariableDeclarationNode decl) {
    Instruction inst;
    IdentifierValue lhs = new IdentifierValue(decl.lhs.value, new TypeValue.fromTypeNode(decl.type));
    
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
  return vst.instructions;
}