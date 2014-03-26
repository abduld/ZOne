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
  
  String histogram = """
      Let hist :: []Integer = 0;
      Map(((x :: Integer) :: Void => x), img);
    """;
  ProgramNode prog = Parse(histogram);
  List<Instruction> insts = Lower(prog);
  List<Instruction> minsts = LiftFunctionPass(insts);
  print(ToJavaScriptCode(minsts));
}
