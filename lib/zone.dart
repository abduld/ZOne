library zone;


import "src/AST/AST.dart";
import "src/Instruction/Instruction.dart";
import "src/Lower/Lower.dart";
import "src/Analysis/Analysis.dart";
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
    
  String histogram2 = """
      data :: []Integer = ReadIntegerList("myFile");
      histogram :: []Integer = Zeros(255);
      Map((x :: Integer) :: Void => histogram[x] += 1; Return ;, data);
  """;
  ProgramNode prog = Parse(testCode);
  List<Instruction> insts = Lower(prog);
  List<InstructionPass> passes = [new LiftFunctionPass(), new PeepholeOptimizePass()];
  List<Instruction> minsts = passes.fold(insts, (List<Instruction> prev, InstructionPass pass) => pass.run(prev));
  print(ToCCode(minsts));
}
