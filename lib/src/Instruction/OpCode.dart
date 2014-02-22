part of zone.instruction;


class OpCode {
  final String opName;
  final int nArgs;

  OpCode(this.opName, this.nArgs);

  get name => opName;
  get unaryQ => nArgs == 1;
  get binaryQ => nArgs == 2;
  
  void visit(InstructionVisitor visitor) =>
      visitor.visitOpCode(this);

  int get hashcode => name.hashcode;
  bool sameQ(other) => hashcode == other.hashcode;
  
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

final OpCode IdentityOp = new UnaryOpCode("Identity");
final OpCode LoadOp = new UnaryOpCode("Load");
final OpCode StoreOp = new UnaryOpCode("Store");
final OpCode AllocaOp = new UnaryOpCode("Alloca");

final OpCode SetOp = new BinaryOpCode("Set");
final OpCode GetOp = new UnaryOpCode("Get");

final OpCode AddOp = new BinaryOpCode("Add");
final OpCode SubtractOp = new BinaryOpCode("Subtract");
final OpCode DivideOp = new BinaryOpCode("Divide");
final OpCode TimesOp = new BinaryOpCode("Times");

final OpCode AddToOp = new UnaryOpCode("AddTo");
final OpCode SubtractFromOp = new UnaryOpCode("SubtractFrom");
final OpCode DivideByOp = new UnaryOpCode("DivideBy");
final OpCode TimesByOp = new UnaryOpCode("TimesBy");

final OpCode CallOp = new OpCode("Call", -1);
final OpCode BranchOp = new OpCode("Branch", 3);

