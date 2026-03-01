#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0: dense, d1: compressed) }>

func.func @tensor_add(%arg0: tensor<32x32xf32, #CSR>,
                      %arg1: tensor<32x32xf32, #CSR>) -> tensor<32x32xf32> attributes {llvm.emit_c_interface} {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<32x32xf32>
  %out = linalg.fill ins(%cst : f32) outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>

  %3 = linalg.add
      ins(%arg0, %arg1 : tensor<32x32xf32, #CSR>, tensor<32x32xf32, #CSR>)
      outs(%out : tensor<32x32xf32>) -> tensor<32x32xf32>

  return %3 : tensor<32x32xf32>
}
