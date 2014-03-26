library zone;


import "src/AST/AST.dart";
import "src/Instruction/Instruction.dart";
import "src/Lower/Lower.dart";
import "src/Backend/Backend.dart";

void RunTests() {
  TestAST();
  
  String testCode = """
      Let k :: Integer = 2 + 32;
      Let k :: Integer = (x :: Integer) :: Integer => Return(x + 1);
      ((x :: Integer) :: Integer => Return(x + 1)) :@ [1,2,3,4];
    """;
  ProgramNode prog = Parse(testCode);
  List<Instruction> insts = Lower(prog);
  List<Instruction> minsts = LiftFunctionPass(insts);
  print(ToJavaScriptCode(minsts));
}
