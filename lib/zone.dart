library zone;


import "src/AST/AST.dart";
import "src/Instruction/Instruction.dart";
import "src/Lower/Lower.dart";

void RunTests() {
  TestAST();
  
  String testCode = """
      Context :foo;
      Let k :: Integer = 2 + 32;
      Let k :: Integer = (x :: Integer) :: Integer => x + 1;
      ((x :: Integer) :: Integer => x + 1) :@ [1,2,3,4];
    """;
  ProgramNode prog = Parse(testCode);
  Lower(prog);
}
