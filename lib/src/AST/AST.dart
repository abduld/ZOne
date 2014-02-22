
library zone.ast;

import 'package:parsers/parsers.dart';
import 'package:unittest/unittest.dart';
import '../Instruction/Instruction.dart';

part './Parser.dart';
part './ASTNode.dart';

ProgramNode Parse(String prog) {
  ZoneParser zoneParser = new ZoneParser();
  List<ASTNode> ast =
      zoneParser.start.parse(prog);
  return new ProgramNode(ast);
}

TestAST() {
  test('start', () {
    String testCode = """
      Context :foo;
      Let k :: Integer = 2 + 32;
      Let k :: Integer = (x :: Integer) :: Integer => x + 1;
      ((x :: Integer) :: Integer => x + 1) :@ [1,2,3,4];
    """;
    print(Parse(testCode));
  });
}