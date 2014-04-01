
part of zone.backend;

class CUDAMemcpyToDevice {
  Value dst;
  Value src;
  int sz;
  CUDAMemcpyToDevice(this.dst, this.src, this.sz);
  String toString() => "cudaMemcpy($dst, $src, $sz, cudaMemcpyHostToDevice)";
}

class CUDAMemcpyToHost {
  Value dst;
  Value src;
  int sz;
  CUDAMemcpyToHost(this.dst, this.src, this.sz);
  String toString() => "cudaMemcpy($dst, $src, $sz, cudaMemcpyDeviceToHost)";
}

class CUDAMalloc {
  Value dst;
  Value sz;
  CUDAMalloc(this.dst, this.sz);
  String toString() => "CUDAMalloc($dst, $sz)";
}

class Dim3 {
  int x, y, z;
  String name;
  Dim3({this.name : "", this.x : 1, this.y : 1, this.z : 1});
}

class CUDABlock extends Dim3 {
  String toString() => "dim3 $name($x, $y, $z)";
}

class CUDAGrid extends Dim3 {
  String toString() => "dim3 $name($x, $y, $z)";
}

class CUDALaunch {
  CUDAGrid grid;
  CUDABlock block;
  List<Value> args;
  Value function;
  
  CUDALaunch(this.grid, this.block, this.function, this.args);
  String toString() {
    if (grid.name.isEmpty) {
      grid.name = this.function.toString() + "gridDim";
    }
    if (block.name.isEmpty) {
      block.name = this.function.toString() + "_blockDim";
    }
    String call = "${function.value}<<<${grid.name}, ${block.name}>>>(${args.map((arg) => arg.accept(this)).join(", ")})";
    return [grid, block, call].join("\n");
  }
}

class CUDAAtomic {
  
}

class CUDAAtomicAdd extends CUDAAtomic {
  
}