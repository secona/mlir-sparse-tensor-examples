#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0: dense, d1: compressed) }>

func.func @tensor_add(%arg0: tensor<?x?xf32, #CSR>) -> tensor<?x?xf32> attributes {llvm.emit_c_interface} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %N = tensor.dim %arg0, %c0 : tensor<?x?xf32, #CSR>
  %M = tensor.dim %arg0, %c1 : tensor<?x?xf32, #CSR>

  %result_out = tensor.empty(%N, %M) : tensor<?x?xf32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(i, j) -> (i, j)>,
      affine_map<(i, j) -> (j, i)>,
      affine_map<(i, j) -> (i, j)>
    ],
    iterator_types = [
      "parallel",
      "parallel"
    ]
  }
  ins(%arg0, %arg0 : tensor<?x?xf32, #CSR>, tensor<?x?xf32, #CSR>)
  outs(%result_out : tensor<?x?xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %sum = arith.addf %a, %b : f32
    linalg.yield %sum : f32
  } -> tensor<?x?xf32>

  return %result : tensor<?x?xf32>
}
