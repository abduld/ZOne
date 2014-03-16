part of zone.instruction;


class OpCode {
  final String opName;
  final int nArgs;

  OpCode(this.opName, this.nArgs);

  String get name => opName;
  bool get unaryQ => nArgs == 1;
  bool get binaryQ => nArgs == 2;
  bool get varArgsQ => nArgs == -1;
  
  Object accept(InstructionVisitor visitor) =>
      visitor.visitOpCode(this);

  int get hashcode => HashCode([name, nArgs]);
  bool sameQ(other) => hashcode == other.hashCode;
  
  String toString() => opName;
}

class UnaryOpCode extends OpCode {
  UnaryOpCode(String opName) : super(opName, 1);
  
  get numArgs => 1;
  get unaryQ => true;
  get binaryQ => false;
}

class BinaryOpCode extends OpCode {
  BinaryOpCode(String opName) : super(opName, 2);
  
  get numArgs => 2;
  get unaryQ => true;
  get binaryQ => true;
}

final OpCode FunctionDefinitionOp = new OpCode("FunctionDefinition", -1);

final OpCode IdentityOp = new UnaryOpCode("Identity");
final OpCode LoadOp = new UnaryOpCode("Load");
final OpCode StoreOp = new UnaryOpCode("Store");
final OpCode AllocaOp = new UnaryOpCode("Alloca");

final OpCode SetOp = new BinaryOpCode("Set");
final OpCode GetOp = new UnaryOpCode("Get");

final OpCode CallOp = new OpCode("Call", -1);
final OpCode LambdaOp = new OpCode("Lambda", -1);
final OpCode ReturnOp = new OpCode("Return", -1);
final OpCode BranchOp = new OpCode("Branch", 3);

