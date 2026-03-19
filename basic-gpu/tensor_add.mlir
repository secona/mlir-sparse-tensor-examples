#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0: dense, d1: compressed) }>

func.func @tensor_add(%arg0: tensor<?x?xf32, #CSR>) -> tensor<?x?xf32> attributes {llvm.emit_c_interface} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %N = tensor.dim %arg0, %c0 : tensor<?x?xf32, #CSR>
  %M = tensor.dim %arg0, %c1 : tensor<?x?xf32, #CSR>

  %dense_a = sparse_tensor.convert %arg0 : tensor<?x?xf32, #CSR> to tensor<?x?xf32>

  %T_out = tensor.empty(%M, %N) : tensor<?x?xf32>
  %T = linalg.transpose
    ins(%dense_a : tensor<?x?xf32>)
    outs(%T_out : tensor<?x?xf32>)
    permutation = [1, 0]

  %result_out = tensor.empty(%N, %M) : tensor<?x?xf32>
  %result = linalg.add
    ins(%dense_a, %T : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%result_out : tensor<?x?xf32>) -> tensor<?x?xf32>

  return %result : tensor<?x?xf32>
}
