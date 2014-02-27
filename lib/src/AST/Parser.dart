part of zone.ast;


final reservedNames = ["Function", "Type", "Macro", "For", "While", "If", "Let", "Context", "True", "False"];

final Parser<String> identStart = letter | oneOf("_+-*?%");
final Parser<String> identLetter = alphanum | oneOf("_?!");

class ZoneParser extends LanguageParsers {

  ZoneParser() : super(reservedNames: reservedNames,
                       commentStart: "/*",
                       commentEnd: "*/",
                       commentLine: "//",
                       identStart: identStart,
                       identLetter: identLetter);
  
  Parser< List<ASTNode> > get start =>
      stmts.between(spaces, eof); 
  
  Parser< List<ASTNode> > get stmts =>
       stmt.endBy(semi)
     | docString;

  Parser<ASTNode> get stmt =>
         contextDeclaration
       | variableDeclaration
       | assignment
       | expr();
  
  Parser< List<ASTNode> > get docString =>
      lexeme(_docString).many ^ (doc) => doc.list;

  Parser get _docString =>
        everythingBetween(string('//'), string('\n'))
      | everythingBetween(string('/*'), string('*/'))
      | everythingBetween(string('/**'), string('*/'));

  Parser<ASTNode> get sym =>
      char(":") + identifier ^ (_, s) => new SymbolNode(s);

  Parser<ASTNode> get contextDeclaration =>
       reserved["Context"]
       + sym
       ^ (_, SymbolNode sym) => new CallNode("Context", [sym]);
  
  Parser<ASTNode> get typeIdentifier =>
       symbol('::') + identifier ^ ((_, t) => new TypeNode(t))
     | symbol('<:') + identifier ^ ((_, t) => new SubTypeNode(t));
  
  Parser<ASTNode> get variableDeclaration =>
      (
          reserved["Let"]
          + identifier
          + typeIdentifier
          + symbol("=")
          + expr()
          ^ (_, id, t, __, e) => new VariableDeclarationNode(t, id, e)
      )
      |
      (
           reserved["Let"]
         + identifier
         + typeIdentifier
         ^ (_, id, t) => new VariableDeclarationNode(t, id)
      );
  
  Parser<ASTNode> get assignment =>
      identifier
      + symbol('=')
      + expr()
      ^ (id, _, val) => new AssignmentNode(id, val);
  
  Parser identity(e) => e;
  
  final longName = {
     "*": "Times",
     "-": "Minus",
     "+": "Plus",
     "/": "Divide",
     "||": "Or",
     "&&": "And",
     "==": "Equal",
     "!=": "NotEqual",
     "<": "LessThan",
     "<=": "LessThanEqual",
     ">": "GreaterThan",
     ">=": "GreaterThanEqual",
     "+=": "AddTo",
     "-=": "SubtractFrom",
     ":@": "Map",
     ":#": "Fold"
  };
  
  _binOp(str) => symbol(str) > success((x, y) => new CallNode(longName[str], [x, y]));

  orExpression() => andExpression().chainl1(_binOp(":@") | _binOp(":#") | _binOp("||"));

  andExpression() => equalityExpression().chainl1(_binOp("&&"));

  equalityExpression() => relationalExpression().chainl1(_binOp("==") | _binOp("!="));

  relationalExpression() => additiveExpression().chainl1(
      _binOp(">") | _binOp(">=") | _binOp("<") | _binOp("<="));

  additiveExpression() => multiplicativeExpression().chainl1(
      _binOp("+") | _binOp("-"));

  multiplicativeExpression() => unaryExpression().chainl1(
      _binOp("*") | _binOp("/") | _binOp("%"));
  
  unaryExpression() => rec(postFixExpression);
  
  postFixExpression() => rec(atom) >> funCalls >> success;
  
  funCalls(prefix) => (funCall(prefix) >> funCalls)
                    | success(prefix);
  
  _args() => parens(rec(expr).sepBy(comma));
  
  funCall(fun) => rec(_args) ^ (arguments) => new CallNode(fun, arguments);

  lambda() => _params() + typeIdentifier + symbol("=>") + rec(expr)
              ^ (parameters, ret, _, body) =>
                  new FunctionDeclarationNode("lambda", ret, parameters, new ProgramNode(body));

  _param() => identifier + typeIdentifier ^ (id, type) => new ParameterNode(id, type);  
  _params() => parens(_param().sepBy(comma));
  
  
  Parser<List<ASTNode>> expr() => rec(orExpression);
  
  Parser<ASTNode> atom() =>
        rec(lambda)
      | rec(variable)
      | rec(literal)
      | parens(rec(expr));
  
  Parser<ASTNode> literal() =>
        intLiteral ^ ((e) => new IntegerNode(e))
      | floatLiteral ^ ((e) => new RealNode(e))
      | stringLiteral ^ ((e) => new StringNode(e))
      | listLiteral ^ ((e) => new ListNode(e))
      | reserved['True'] ^ ((_) => new TrueNode())
      | reserved['False'] ^ ((_) => new FalseNode());
  
  Parser<ASTNode> get listLiteral =>
      brackets(atom().sepBy(comma));
  
  Parser<ASTNode> variable() =>
      identifier ^ ((e) => new IdentifierNode(e));
   
}

