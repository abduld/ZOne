library zone;


import "src/AST/AST.dart";
import "src/Instruction/Instruction.dart";
import "src/Lower/Lower.dart";
import "src/Backend/Backend.dart";

void RunTests() {
  TestAST();
  
  String testCode = """
      Let k :: Integer = 2 + 32;
      Let g :: Integer = 2 + k;
      Let m :: Integer = 2 + g;
      Let i :: Integer = 2 + m;
      //Let k :: Integer = (x :: Integer) :: Integer => Return(x + 1);
      Map((x :: Integer) :: Integer => Return(x + 1), [1,2,3,4]);
      //((x :: Integer) :: Integer => Return(x + 1)) :@ [1,2,3,4];
    """;
  
  String histogram = """
      Let hist :: []Integer = 0;
      Map(((x :: Integer) :: Void => x), img);
    """;
  ProgramNode prog = Parse(testCode);
  List<Instruction> insts = Lower(prog);
  List<InstructionPass> passes = [new LiftFunctionPass(), new PeepholeOptimizePass()];
  List<Instruction> minsts = passes.fold(insts, (List<Instruction> prev, InstructionPass pass) => pass.run(prev));
  print(ToCCode(minsts));
}
