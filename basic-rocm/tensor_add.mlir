#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0: dense, d1: compressed) }>

func.func @tensor_add(%arg0: tensor<32x32xf32, #CSR>) -> tensor<32x32xf32> attributes {llvm.emit_c_interface} {
  %out = tensor.empty() : tensor<32x32xf32>

  %T_out = tensor.empty() : tensor<32x32xf32, #CSR>
  %T = linalg.transpose
    ins(%arg0 : tensor<32x32xf32, #CSR>)
    outs(%T_out : tensor<32x32xf32, #CSR>)
    permutation = [1, 0]

  %result = linalg.add
    ins(%arg0, %T : tensor<32x32xf32, #CSR>, tensor<32x32xf32, #CSR>)
    outs(%out : tensor<32x32xf32>) -> tensor<32x32xf32>

  return %result : tensor<32x32xf32>
}
