func.func @vector_add_gpu(%A: memref<?xf32>,
                          %B: memref<?xf32>,
                          %C: memref<?xf32>,
                          %num_elements: index) attributes {llvm.emit_c_interface} {
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index

  gpu.launch
    blocks(%bx, %by, %bz) in (%grid_x = %c256, %grid_y = %c1, %grid_z = %c1)
    threads(%tx, %ty, %tz) in (%block_x = %c256, %block_y = %c1, %block_z = %c1) {

    %b_dim_x = gpu.block_dim x
    %offset = arith.muli %bx, %b_dim_x : index
    %i = arith.addi %offset, %tx : index

    %in_bounds = arith.cmpi slt, %i, %num_elements : index
    scf.if %in_bounds {
      %a_val = memref.load %A[%i] : memref<?xf32>
      %b_val = memref.load %B[%i] : memref<?xf32>
      %c_val = arith.addf %a_val, %b_val : f32
      memref.store %c_val, %C[%i] : memref<?xf32>
    }

    gpu.terminator
  }

  return
}
