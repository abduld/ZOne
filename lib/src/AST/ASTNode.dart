
part of zone.ast;

abstract class ASTNodeVisitor {
  Object visitProgramNode(ProgramNode prog);
  Object visitDocumentationNode(DocumentationNode doc);
  Object visitCallNode(CallNode call);
  Object visitAtomNode(AtomNode atm);
  Object visitTypeNode(TypeNode typ);
  Object visitVariableDeclarationNode(VariableDeclarationNode decl);
  Object visitAssignmentNode(AssignmentNode ass);
  Object visitFunctionDeclarationNode(FunctionDeclarationNode fundecl);
  Object visitParameterNode(ParameterNode param);
}

abstract class ASTNode {
  String nodeType;
  ASTNode(this.nodeType);

  Object accept(ASTNodeVisitor visitor);
  List<Object> visitChildren(ASTNodeVisitor visitor);

  String toString() => "ASTNode($nodeType)";
}

class ProgramNode extends ASTNode {
  List<ASTNode> body;

  ProgramNode(body0) : super("Program") {
    if (body0 is List) {
      body = body0;
    } else {
      body = [body0];
    }
  }

  String toString() => body.join('\n');

  Object accept(ASTNodeVisitor visitor) {
    return visitor.visitProgramNode(this);
  }

  List<Object> visitChildren(ASTNodeVisitor visitor) {
    List<Object> lst = new List<Object>();
    body.forEach((stmt) => lst.add(stmt.accept(visitor)));
    return lst;
  }
}

class DocumentationNode extends ASTNode {
  List<String> doc;

  DocumentationNode(this.doc) : super("Documentation");

  Object accept(ASTNodeVisitor visitor) {
    return visitor.visitDocumentationNode(this);
  }

  List<Object> visitChildren(ASTNodeVisitor visitor) => [];

  String toString() => "$nodeType[$doc]";
}

abstract class ExprNode extends ASTNode {
  ExprNode(String s) : super(s);

  Object accept(ASTNodeVisitor visitor);
  List<Object> visitChildren(ASTNodeVisitor visitor);
}

class AtomNode<T> extends ExprNode {
  T value;
  AtomNode(String name, this.value) : super("Atom");
  String toString() => "$value";

  Object accept(ASTNodeVisitor visitor) {
    return visitor.visitAtomNode(this);
  }

  List<Object> visitChildren(ASTNodeVisitor visitor) => [];
}

class SymbolNode extends AtomNode<String> {
  SymbolNode(String name) : super("Symbol", name);
}

class IntegerNode extends AtomNode<int> {
  IntegerNode(int val) : super("Integer", val);
}

class RealNode extends AtomNode<double> {
  RealNode(double val) : super("Real", val);
}

class StringNode extends AtomNode<String> {
  StringNode(String val) : super("String", val);
}

class ListNode extends AtomNode<List<AtomNode>> {
  ListNode(List<AtomNode> val) : super("List", val);
}

class BooleanNode extends AtomNode<bool> {
  BooleanNode(bool val) : super("Boolean", val);
}

class TrueNode extends BooleanNode {
  TrueNode() : super(true);
}

class FalseNode extends BooleanNode {
  FalseNode() : super(false);
}

class IdentifierNode extends AtomNode<String> {
  IdentifierNode(id) : super("Identifier", id.toString());
}

class SubTypeNode extends ASTNode {
  String type;
  SubTypeNode(this.type) : super("Type");
  String toString() => "<: $type";

  Object accept(ASTNodeVisitor visitor) {
    return visitor.visitTypeNode(this);
  }

  List<Object> visitChildren(ASTNodeVisitor visitor) => [];
}

class TypeNode extends SubTypeNode {
  TypeNode(String type) : super(type);
  String toString() => ":: $type";
  TypeValue toTypeValue() => new TypeValue(type);
}

class VariableDeclarationNode extends ASTNode {
  IdentifierNode id;
  TypeNode type;
  ASTNode e;

  VariableDeclarationNode(this.type, String name, [this.e]) : super("VariableDeclaration") {
    id = new IdentifierNode(name);
  }

  String toString() {
    if (hasRHS) {
      return "Let $id $type";
    } else {
      return "Let $id $type = $e";
    }
  }

  IdentifierNode get lhs => id;
  ASTNode get rhs => e;

  bool get hasRHS => e != null;

  Object accept(ASTNodeVisitor visitor) {
    return visitor.visitVariableDeclarationNode(this);
  }

  List<Object> visitChildren(ASTNodeVisitor visitor) {
    List<Object> lst = new List<Object>();

    lst.add(id.accept(visitor));
    lst.add(type.accept(visitor));

    if (hasRHS) {
      lst.add(e.accept(visitor));
    }
    return lst;
  }
}

class AssignmentNode extends ASTNode {
  IdentifierNode id;
  ASTNode rhs;

  AssignmentNode(String name, this.rhs) : super("AssignmentNode") {
    id = new IdentifierNode(name);
  }

  String toString() => "$id = $rhs";

  Object accept(ASTNodeVisitor visitor) {
    return visitor.visitAssignmentNode(this);
  }

  List<Object> visitChildren(ASTNodeVisitor visitor) {
    return [id.accept(visitor),
    rhs.accept(visitor)];
  }
}

class CallNode extends ExprNode {
  IdentifierNode f;
  List<ExprNode> args;

  CallNode(fname, this.args) : super("Call") {
    if (fname is IdentifierNode) {
      f = fname;
    } else {
      f = new IdentifierNode(fname);
    }
  }

  String get name => f.value;

  String toString() {
    String s = args.join(', ');
    return "$f($s)";
  }

  Object accept(ASTNodeVisitor visitor) {
    return visitor.visitCallNode(this);
  }

  List<Object> visitChildren(ASTNodeVisitor visitor) {
    List<Object> lst = [f.accept(visitor)];
    args.map((arg) => lst.add(arg.accept(visitor)));
    return lst;
  }
}

class FunctionDeclarationNode extends ExprNode {
  IdentifierNode id;
  List<ParameterNode> args;
  TypeNode returnType;
  ASTNode body;
  FunctionDeclarationNode(String name, this.returnType, this.args, this.body) : super("Function") {
    id = new IdentifierNode(name);
  }
  String toString() {
    String s = args.join(', ');
    return "$id ($s) $returnType => $body";
  }

  Object accept(ASTNodeVisitor visitor) {
    return visitor.visitFunctionDeclarationNode(this);
  }

  List<Object> visitChildren(ASTNodeVisitor visitor) {
    return body.accept(visitor);
  }
}

class ParameterNode extends ASTNode {
  IdentifierNode id;
  TypeNode type;

  ParameterNode(name, this.type) : super("Parameter") {
    id = new IdentifierNode(name);
  }

  String toString() => "$id $type";

  Object accept(ASTNodeVisitor visitor) {
    return visitor.visitParameterNode(this);
  }

  List<Object> visitChildren(ASTNodeVisitor visitor) {
    return [id.accept(visitor),
    type.accept(visitor)];
  }
}


