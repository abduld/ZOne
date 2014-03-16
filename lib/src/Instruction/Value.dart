part of zone.instruction;

abstract class Value<T> {
  T value;
  
  Value(this.value);

  TypeValue get type => 
      (new SystemType("UnknownType")).value;

  int get hashCode => HashCode([value]);
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


class IdentifierValue extends Value<String> {
  TypeValue typ;
  IdentifierValue(String name, [this.typ]) : super(name) {
    if (typ == null) {
      typ = (new SystemType("UnknownType")).value;
    }
  }
  TypeValue get type => typ;
}

class SymbolValue extends Value<String> {
  SymbolValue(String name) : super(name);
  TypeValue get type => (new SystemType("SymbolType")).value;
}

class StringValue extends Value<String> {
  StringValue(String name) : super(name);
  TypeValue get type => (new SystemType("StringType")).value;
}

class ListValue extends Value<List<Value>> {
  ListValue(List<Value> lst) : super(lst);
  TypeValue get type => (new SystemType("ListType")).value;
}

class IntegerValue extends Value<int> {
  IntegerValue(int ii) : super(ii);
  TypeValue get type => (new SystemType("IntegerType")).value;
}

class RealValue extends Value<double> {
  RealValue(double dd) : super(dd);
  TypeValue get type => (new SystemType("RealType")).value;
}

class TrueValue extends Value<bool> {
  TrueValue() : super(true);
  TypeValue get type => (new SystemType("BooleanType")).value;
}

class FalseValue extends Value<bool> {
  FalseValue() : super(false);
  TypeValue get type => (new SystemType("BooleanType")).value;
}

class UnknownValue extends Value<String> {
  UnknownValue(String s) : super(s);
  TypeValue get type => (new SystemType("UnknownType")).value;
}


class ValueFactory {
  final Value val;
  static Map<int, ValueFactory> _cache;

    factory ValueFactory(Value val) {
      if (_cache == null) {
        _cache = {};
      }

      if (_cache.containsKey(val.hashCode)) {
        return _cache[val.hashCode];
      } else {
        final v = new ValueFactory._internal(val);
        _cache[val.hashCode] = v;
        return v;
      }
    }

    ValueFactory._internal(this.val);
    Value get value => val;
    
  }

class SystemType {
  final TypeValue val;
  static Map<String, TypeValue> SystemTypes = {};
  
  factory SystemType(String sym) {
    TypeValue s = new TypeValue(sym);
    ValueFactory vf = new ValueFactory(s);
    final v = new SystemType._internal(vf.value);
    SystemTypes[sym] = v;
    return v;
  }
  SystemType._internal(this.val);
  TypeValue get value => val;
  bool containsQ(TypeValue val) =>
    SystemTypes.containsKey(val.value);
}

class SystemSymbol {
  final SymbolValue val;
  static Map<String, SymbolValue> SystemSymbols = {};
  
  factory SystemSymbol(String sym) {
    SymbolValue s = new SymbolValue(sym);
    ValueFactory vf = new ValueFactory(s);
    final v = new SystemSymbol._internal(vf.value);
    SystemSymbols[sym] = v;
    return v;
  }
  SystemSymbol._internal(this.val);
  SymbolValue get value => val;
  bool containsQ(SymbolValue val) =>
      SystemSymbols.containsKey(val.value);
}

bool SystemTypeQ(val) {
    if (_SystemTypes.contains(val)) {
      new SystemType(val);
      return true;
    } else if (val is String) {
      return SystemTypeQ(new TypeValue(val));
    } else if (val is TypeValue){
      return SystemType.SystemTypes.containsKey(val.value);
    }
  return false;
}

bool SystemSymbolQ(val) {
    if (_SystemSymbols.contains(val)) {
      new SystemSymbol(val);
      return true;
    } else if (val is String) {
      return SystemSymbolQ(new SymbolValue(val));
    } else if (val is SymbolValue){
      return SystemSymbol.SystemSymbols.containsKey(val.value);
    }
  return false;
}


final List<String> _SystemSymbols = [
  "Context", "Plus", "Minus", "Map", "Reduce"                                    
];

final List<String> _SystemTypes = [
  "UnknownType", "IntegerType", "RealType", "BooleanType", "StringType",
  "ListType", "SymbolType"
];
