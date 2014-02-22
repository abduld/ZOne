
library zone.utilities;

int HashValues(List elems) {
  int hash = 0xDEAD;
  
  int combine(int hash, int value) {
    hash = 0x1fffffff & (hash + value);
    hash = 0x1fffffff & (hash + ((0x0007ffff & hash) << 10));
    return hash ^ (hash >> 6);
  }
  
  int finish(int hash) {
    hash = 0x1fffffff & (hash + ((0x03ffffff & hash) <<  3));
    hash = hash ^ (hash >> 11);
    return 0x1fffffff & (hash + ((0x00003fff & hash) << 15));
  }
  
  elems.map((elem) => hash = combine(hash, elem));
  return finish(hash);
}

int HashCode(List elems) {
  return HashCode(elems.map((elem) => elem.hashCode));
}