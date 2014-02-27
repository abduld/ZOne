part of zone.instruction;

abstract class Value<T> {
  T value;
  
  Value(this.value);

  TypeValue get type => UnknownType;

  int get hashCode => HashCode([value, type]);
  bool sameQ(other) => hashCode == other.hashCode;

  Object accept(InstructionVisitor visitor) =>
      visitor.visitValue(this);
  
  String toString() => value.toString();
}

class SubTypeValue extends Value<List<String>> {
  SubTypeValue(String typ) : super([typ]);
  String toString() => value.join(' :: ');
  SubTypeValue.fromSubTypeNode(SubTypeNode nd) :
    super([nd.type]);
}

class TypeValue extends SubTypeValue {
  TypeValue(type) : super(type);
  TypeValue.fromTypeNode(TypeNode nd) :
    super(nd.type);
}

final TypeValue UnknownType = new TypeValue("Unknown");
final TypeValue IntegerType = new TypeValue("Integer");
final TypeValue RealType = new TypeValue("RealType");
final TypeValue BoolType = new TypeValue("BoolType");
final TypeValue StringType = new TypeValue("String");
final TypeValue ListType = new TypeValue("ListType");
final TypeValue SymbolType = new TypeValue("Symbol");

class IdentifierValue extends Value<String> {
  TypeValue typ;
  IdentifierValue(String name, [this.typ]) : super(name) {
    if (typ == null) {
      typ = UnknownType;
    }
  }
  TypeValue get type => typ;
}

class SymbolValue extends Value<String> {
  SymbolValue(String name) : super(name);
  TypeValue get type => SymbolType;
}

class StringValue extends Value<String> {
  StringValue(String name) : super(name);
  TypeValue get type => StringType;
}

class ListValue extends Value<List<Value>> {
  ListValue(List<Value> lst) : super(lst);
  TypeValue get type => ListType;
}

class IntegerValue extends Value<int> {
  IntegerValue(int ii) : super(ii);
  TypeValue get type => IntegerType;
}

class RealValue extends Value<double> {
  RealValue(double dd) : super(dd);
  TypeValue get type => RealType;
}

class TrueValue extends Value<bool> {
  TrueValue() : super(true);
  TypeValue get type => BoolType;
}

class FalseValue extends Value<bool> {
  FalseValue() : super(false);
  TypeValue get type => BoolType;
}

class UnknownValue extends Value<String> {
  UnknownValue(String s) : super(s);
  TypeValue get type => UnknownType;
}

final SymbolValue MapSymbol = new SymbolValue("Map");
final SymbolValue ReduceSymbol = new SymbolValue("Reduce");
