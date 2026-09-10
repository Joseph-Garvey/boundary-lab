/*
 * ABI probe for the Accelerate sparse-solver structs.
 *
 * BeatEngineAccelerateSparse.jl mirrors these C types by hand, and a layout
 * mistake there is silent memory corruption rather than a compile failure. This
 * program prints the sizes, offsets and enum values that
 * `accelerate_sparse_abi_matches_reference()` asserts. Re-run it after an SDK
 * update, and update the Julia reference values if anything moved:
 *
 *     clang -mmacosx-version-min=15.5 -Wno-unguarded-availability-new \
 *         -framework Accelerate -o /tmp/probe scripts/probe-accelerate-sparse-abi.c
 *     /tmp/probe
 */
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
int main(void){
#define S(t) printf("sizeof(%-42s) = %3zu\n", #t, sizeof(t))
#define O(t,f) printf("offsetof(%-30s, %-26s) = %3zu\n", #t, #f, offsetof(t,f))
  S(SparseAttributesComplex_t);
  S(SparseMatrixStructureComplex);
  S(SparseMatrix_Complex_Float);
  S(DenseMatrix_Complex_Float);
  S(SparseSymbolicFactorOptions);
  S(SparseNumericFactorOptions);
  S(SparseOpaqueSymbolicFactorization);
  S(SparseOpaqueFactorization_Complex_Float);
  puts("");
  O(SparseMatrixStructureComplex, rowCount);
  O(SparseMatrixStructureComplex, columnCount);
  O(SparseMatrixStructureComplex, columnStarts);
  O(SparseMatrixStructureComplex, rowIndices);
  O(SparseMatrixStructureComplex, attributes);
  O(SparseMatrixStructureComplex, blockSize);
  O(SparseMatrix_Complex_Float, data);
  puts("");
  O(DenseMatrix_Complex_Float, rowCount);
  O(DenseMatrix_Complex_Float, columnCount);
  O(DenseMatrix_Complex_Float, columnStride);
  O(DenseMatrix_Complex_Float, attributes);
  O(DenseMatrix_Complex_Float, data);
  puts("");
  O(SparseSymbolicFactorOptions, control);
  O(SparseSymbolicFactorOptions, orderMethod);
  O(SparseSymbolicFactorOptions, order);
  O(SparseSymbolicFactorOptions, ignoreRowsAndColumns);
  O(SparseSymbolicFactorOptions, malloc);
  O(SparseSymbolicFactorOptions, free);
  O(SparseSymbolicFactorOptions, reportError);
  puts("");
  O(SparseNumericFactorOptions, control);
  O(SparseNumericFactorOptions, scalingMethod);
  O(SparseNumericFactorOptions, scaling);
  O(SparseNumericFactorOptions, pivotTolerance);
  O(SparseNumericFactorOptions, zeroTolerance);
  puts("");
  O(SparseOpaqueSymbolicFactorization, status);
  O(SparseOpaqueSymbolicFactorization, rowCount);
  O(SparseOpaqueSymbolicFactorization, columnCount);
  O(SparseOpaqueSymbolicFactorization, attributes);
  O(SparseOpaqueSymbolicFactorization, blockSize);
  O(SparseOpaqueSymbolicFactorization, type);
  O(SparseOpaqueSymbolicFactorization, factorization);
  O(SparseOpaqueSymbolicFactorization, workspaceSize_Float);
  O(SparseOpaqueSymbolicFactorization, factorSize_Float);
  puts("");
  O(SparseOpaqueFactorization_Complex_Float, status);
  O(SparseOpaqueFactorization_Complex_Float, attributes);
  O(SparseOpaqueFactorization_Complex_Float, symbolicFactorization);
  O(SparseOpaqueFactorization_Complex_Float, userFactorStorage);
  O(SparseOpaqueFactorization_Complex_Float, numericFactorization);
  O(SparseOpaqueFactorization_Complex_Float, solveWorkspaceRequiredStatic);
  O(SparseOpaqueFactorization_Complex_Float, solveWorkspaceRequiredPerRHS);
  puts("");
  printf("SparseFactorizationLU = %d\n", (int)SparseFactorizationLU);
  printf("SparseOrdinary        = %d\n", (int)SparseOrdinary);
  printf("SparseStatusOK        = %d\n", (int)SparseStatusOK);
  { SparseAttributesComplex_t a = {0}; a.kind = SparseOrdinary;
    unsigned short raw; memcpy(&raw,&a,2);
    printf("attributes(ordinary) raw = 0x%04x\n", raw); }
  return 0;
}
