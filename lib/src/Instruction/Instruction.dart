
library zone.instruction;

import '../Utilities/Utilities.dart';
import '../AST/AST.dart';

part './OpCode.dart';
part './Value.dart';


abstract class InstructionVisitor {
  Object visitOpCode(OpCode op);
  Object visitValue(Value val);
  Object visitEmptyInstruction(EmptyInstruction inst);
  Object visitUnaryInstruction(UnaryInstruction inst);
  Object visitBinaryInstruction(BinaryInstruction inst);
  Object visitCallInstruction(CallInstruction inst);
  Object visitLambdaInstruction(LambdaInstruction inst);
  Object visitReturnInstruction(ReturnInstruction inst);
  Object visitBranchInstruction(BranchInstruction inst);
  Object visitMapInstruction(MapInstruction inst);
  Object visitReduceInstruction(ReduceInstruction inst);
}


abstract class Instruction {
  OpCode op;
  Value target;
  List<Value> args;
  
  Instruction(this.op, this.target, this.args);
  
  String toString() {
    String params = args.join(', ');
    if (target == null) {
      return "$op $params";
    } else {
      return "$target = $op $params";
    }
  }
  
  int get nargs => args == null ? 0 : args.length;

  int get hashcode => HashCode([op, target, args]);
  bool sameQ(other) => hashcode == other.hashcode;

  Object accept(InstructionVisitor visitor);
  List<Object> visitChildren(InstructionVisitor visitor) {
    if (target != null) {
      target.accept(visitor);
    }
    return args.map((arg) => arg.accept(visitor));
  }
}

class EmptyInstruction extends Instruction {
  EmptyInstruction(OpCode op, Value target) :
    super(op, target, []);

  Object accept(InstructionVisitor visitor) =>
      visitor.visitEmptyInstruction(this);
}

class UnaryInstruction extends Instruction {
  UnaryInstruction(OpCode op, Value target, Value arg) :
    super(op, target, [arg]);

  Object accept(InstructionVisitor visitor) =>
      visitor.visitUnaryInstruction(this);
}

class BinaryInstruction extends Instruction {
  BinaryInstruction(OpCode op, Value target, Value arg1, Value arg2) :
    super(op, target, [arg1, arg2]);
  
  Object accept(InstructionVisitor visitor) =>
      visitor.visitBinaryInstruction(this);
}

class BranchInstruction extends Instruction {
  BranchInstruction(List<Value> args) : super(BranchOp, null, args);

  Object accept(InstructionVisitor visitor) =>
      visitor.visitBranchInstruction(this);
}

class CallInstruction extends Instruction {
  CallInstruction(Value lhs, Value name, List<Value> args) :
    super(CallOp, lhs, args..insert(0, name));

  Object accept(InstructionVisitor visitor) =>
      visitor.visitCallInstruction(this);
}

class LambdaInstruction extends Instruction {
  List<Instruction> body = [];
  
  LambdaInstruction(Value lhs, List<Value> args, this.body) :
    super(LambdaOp, lhs, args);

  Object accept(InstructionVisitor visitor) =>
      visitor.visitLambdaInstruction(this);

  String toString() =>
    '$target = lambda(${args.join(", ")}) => ($body)';
}

class ReturnInstruction extends Instruction {
  ReturnInstruction(Value arg) :
    super(ReturnOp, null, [arg]);

  Object accept(InstructionVisitor visitor) =>
      visitor.visitReturnInstruction(this);
}

class MapInstruction extends CallInstruction {
  MapInstruction(Value lhs, List<Value> args) :
    super(MapSymbol, lhs, args);

  Object accept(InstructionVisitor visitor) =>
      visitor.visitMapInstruction(this);
}

class ReduceInstruction extends CallInstruction {
  ReduceInstruction(Value lhs, List<Value> args) :
    super(ReduceSymbol, lhs, args);

  Object accept(InstructionVisitor visitor) =>
      visitor.visitReduceInstruction(this);
}
