"""
Apple Accelerate sparse LU bindings.

The FEM static condensation spends nearly all of its time in sparse triangular
solves against the interior factorization — 98.6% of the Schur complement stage
on the `F2B_FLH` fixture. Accelerate's Sparse Solvers offer a complex LU with a
first-class multi-RHS solve, in single precision, which matches the rest of the
FP32 Metal pipeline where the UMFPACK path forces `ComplexF64`.

This module is a direct `ccall` binding, so every struct below mirrors a C type
from `<Accelerate/Accelerate.h>`. The layouts were taken from a C probe rather
than by eye — see `scripts/probe-accelerate-sparse-abi.c`, which prints the
`sizeof` and `offsetof` values these declarations must reproduce, and
`accelerate_sparse_abi_matches_reference()` below, which asserts them.

A layout error here is silent memory corruption, not a compile failure, so the
`accelerate_sparse_selftest()` entry point checks a factorization against a
known system before any of this is trusted with a real solve.
"""
module BeatEngineAccelerateSparse

using LinearAlgebra, Random, SparseArrays

export AccelerateSparseLU,
    accelerate_sparse_available,
    accelerate_sparse_lu,
    accelerate_sparse_refactor!,
    accelerate_sparse_solve!,
    accelerate_sparse_free!,
    accelerate_sparse_selftest,
    accelerate_sparse_abi_matches_reference

const ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Accelerate"

# SPARSE_ENUM underlying types, from Solve.h: SparseFactorization is uint8_t,
# SparseControl uint32_t, SparseOrder and SparseScaling uint8_t, SparseStatus int.
const SPARSE_FACTORIZATION_LU = UInt8(80)
const SPARSE_STATUS_OK = Int32(0)
const SPARSE_DEFAULT_CONTROL = UInt32(0)
const SPARSE_ORDER_DEFAULT = UInt8(0)
const SPARSE_SCALING_DEFAULT = UInt8(0)
# SparseAttributesComplex_t is a 4-byte bitfield. All-zero means an ordinary,
# non-transposed matrix, which is the only configuration used here.
const SPARSE_ATTRIBUTES_ORDINARY = UInt32(0)

# `_SparseRefactorLU` is a macOS 15.5 addition, so `accelerate_sparse_refactor!`
# is only reachable there. The one-shot `accelerate_sparse_lu` path predates it
# and is what the condensation currently uses.
struct SparseMatrixStructureComplex
    row_count::Cint
    column_count::Cint
    column_starts::Ptr{Clong}
    row_indices::Ptr{Cint}
    attributes::UInt32
    block_size::UInt8
end

struct SparseMatrixComplex{C}
    structure::SparseMatrixStructureComplex
    data::Ptr{C}
end

struct DenseMatrixComplex{C}
    row_count::Cint
    column_count::Cint
    column_stride::Cint
    attributes::UInt32
    data::Ptr{C}
end

struct SparseSymbolicFactorOptions
    control::UInt32
    order_method::UInt8
    order::Ptr{Cint}
    ignore_rows_and_columns::Ptr{Cint}
    malloc::Ptr{Cvoid}
    free::Ptr{Cvoid}
    report_error::Ptr{Cvoid}
end

struct SparseNumericFactorOptions
    control::UInt32
    scaling_method::UInt8
    scaling::Ptr{Cvoid}
    pivot_tolerance::Cdouble
    zero_tolerance::Cdouble
end

struct SparseOpaqueSymbolicFactorization
    status::Int32
    row_count::Cint
    column_count::Cint
    attributes::UInt32
    block_size::UInt8
    type::UInt8
    factorization::Ptr{Cvoid}
    workspace_size_float::Csize_t
    workspace_size_double::Csize_t
    factor_size_float::Csize_t
    factor_size_double::Csize_t
end

struct SparseOpaqueFactorizationComplex
    status::Int32
    attributes::UInt32
    symbolic_factorization::SparseOpaqueSymbolicFactorization
    user_factor_storage::Bool
    numeric_factorization::Ptr{Cvoid}
    solve_workspace_required_static::Csize_t
    solve_workspace_required_per_rhs::Csize_t
end

"""
    accelerate_sparse_abi_matches_reference() -> Bool

Check the Julia struct declarations against the sizes and offsets the C probe
reported on macOS 15.7 / arm64. These are the numbers a layout mistake would
break, and they are cheap enough to assert on every load.
"""
function accelerate_sparse_abi_matches_reference()
    expected_sizes = (
        (SparseMatrixStructureComplex, 32),
        (SparseMatrixComplex{ComplexF32}, 40),
        (SparseMatrixComplex{ComplexF64}, 40),
        (DenseMatrixComplex{ComplexF32}, 24),
        (DenseMatrixComplex{ComplexF64}, 24),
        (SparseSymbolicFactorOptions, 48),
        (SparseNumericFactorOptions, 32),
        (SparseOpaqueSymbolicFactorization, 64),
        (SparseOpaqueFactorizationComplex, 104),
    )
    for (type, size) in expected_sizes
        sizeof(type) == size || return false
    end
    expected_offsets = (
        (SparseMatrixStructureComplex, :column_starts, 8),
        (SparseMatrixStructureComplex, :row_indices, 16),
        (SparseMatrixStructureComplex, :attributes, 24),
        (SparseMatrixStructureComplex, :block_size, 28),
        (SparseMatrixComplex{ComplexF32}, :data, 32),
        (SparseMatrixComplex{ComplexF64}, :data, 32),
        (DenseMatrixComplex{ComplexF32}, :attributes, 12),
        (DenseMatrixComplex{ComplexF32}, :data, 16),
        (DenseMatrixComplex{ComplexF64}, :data, 16),
        (SparseSymbolicFactorOptions, :order, 8),
        (SparseSymbolicFactorOptions, :malloc, 24),
        (SparseSymbolicFactorOptions, :report_error, 40),
        (SparseNumericFactorOptions, :pivot_tolerance, 16),
        (SparseNumericFactorOptions, :zero_tolerance, 24),
        (SparseOpaqueSymbolicFactorization, :block_size, 16),
        (SparseOpaqueSymbolicFactorization, :type, 17),
        (SparseOpaqueSymbolicFactorization, :factorization, 24),
        (SparseOpaqueFactorizationComplex, :symbolic_factorization, 8),
        (SparseOpaqueFactorizationComplex, :user_factor_storage, 72),
        (SparseOpaqueFactorizationComplex, :numeric_factorization, 80),
        (SparseOpaqueFactorizationComplex, :solve_workspace_required_per_rhs, 96),
    )
    for (type, field, offset) in expected_offsets
        index = findfirst(==(field), fieldnames(type))
        isnothing(index) && return false
        fieldoffset(type, index) == offset || return false
    end
    return true
end

_default_symbolic_options() = SparseSymbolicFactorOptions(
    SPARSE_DEFAULT_CONTROL,
    SPARSE_ORDER_DEFAULT,
    Ptr{Cint}(C_NULL),
    Ptr{Cint}(C_NULL),
    cglobal(:malloc),
    cglobal(:free),
    Ptr{Cvoid}(C_NULL),
)

# Matches _SparseDefaultNumericFactorOptions_Complex_Float and _Complex_Double in
# SolveImplementation.h: the pivot tolerance recommended there for difficult
# matrices at each precision, and a zero tolerance a few orders of magnitude
# below that precision's epsilon.
_default_numeric_options(::Type{ComplexF32}) = SparseNumericFactorOptions(
    SPARSE_DEFAULT_CONTROL,
    SPARSE_SCALING_DEFAULT,
    Ptr{Cvoid}(C_NULL),
    0.1,
    1.0e-4 * eps(Float32),
)

_default_numeric_options(::Type{ComplexF64}) = SparseNumericFactorOptions(
    SPARSE_DEFAULT_CONTROL,
    SPARSE_SCALING_DEFAULT,
    Ptr{Cvoid}(C_NULL),
    0.01,
    1.0e-4 * eps(Float64),
)

# Accelerate exposes one symbol per element type rather than a generic entry
# point, and `ccall` needs a literal symbol, so the four raw entry points are
# generated once per precision below.

"""
A live Accelerate LU factorization, plus the CSC buffers it borrows.

Accelerate does not copy the structure or value arrays, so `column_starts`,
`row_indices`, and `values` must outlive the factorization. Holding them on the
struct is what keeps them from being collected.
"""
mutable struct AccelerateSparseLU{C<:Union{ComplexF32,ComplexF64}}
    factorization::SparseOpaqueFactorizationComplex
    column_starts::Vector{Clong}
    row_indices::Vector{Cint}
    values::Vector{C}
    size::Int
    freed::Bool
end

Base.eltype(::AccelerateSparseLU{C}) where {C} = C

accelerate_sparse_available() =
    Sys.isapple() && Sys.ARCH === :aarch64 && accelerate_sparse_abi_matches_reference()


for (element, suffix) in ((:ComplexF32, "Complex_Float"), (:ComplexF64, "Complex_Double"))
    @eval begin
        function _raw_factor_lu(matrix::SparseMatrixComplex{$element}, symbolic, numeric)
            return ccall(
                ($(QuoteNode(Symbol("_SparseFactorLU_", suffix))), ACCELERATE),
                SparseOpaqueFactorizationComplex,
                (
                    UInt8,
                    Ref{SparseMatrixComplex{$element}},
                    Ref{SparseSymbolicFactorOptions},
                    Ref{SparseNumericFactorOptions},
                ),
                SPARSE_FACTORIZATION_LU,
                matrix,
                symbolic,
                numeric,
            )
        end

        function _raw_refactor_lu(
            matrix::SparseMatrixComplex{$element},
            factorization::Ref{SparseOpaqueFactorizationComplex},
            numeric,
            workspace,
        )
            return ccall(
                ($(QuoteNode(Symbol("_SparseRefactorLU_", suffix))), ACCELERATE),
                Cvoid,
                (
                    Ref{SparseMatrixComplex{$element}},
                    Ref{SparseOpaqueFactorizationComplex},
                    Ref{SparseNumericFactorOptions},
                    Ptr{Cvoid},
                ),
                matrix,
                factorization,
                numeric,
                workspace,
            )
        end

        function _raw_solve(
            factorization::Ref{SparseOpaqueFactorizationComplex},
            b::DenseMatrixComplex{$element},
            x::DenseMatrixComplex{$element},
            workspace,
        )
            return ccall(
                ($(QuoteNode(Symbol("_SparseSolveOpaque_", suffix))), ACCELERATE),
                Cvoid,
                (
                    Ref{SparseOpaqueFactorizationComplex},
                    Ref{DenseMatrixComplex{$element}},
                    Ref{DenseMatrixComplex{$element}},
                    Ptr{Cvoid},
                ),
                factorization,
                b,
                x,
                workspace,
            )
        end

        function _raw_destroy(
            ::Type{$element},
            factorization::Ref{SparseOpaqueFactorizationComplex},
        )
            return ccall(
                ($(QuoteNode(Symbol("_SparseDestroyOpaqueNumeric_", suffix))), ACCELERATE),
                Cvoid,
                (Ref{SparseOpaqueFactorizationComplex},),
                factorization,
            )
        end
    end
end

function _matrix_handle(lu::AccelerateSparseLU{C}) where {C}
    structure = SparseMatrixStructureComplex(
        Cint(lu.size),
        Cint(lu.size),
        pointer(lu.column_starts),
        pointer(lu.row_indices),
        SPARSE_ATTRIBUTES_ORDINARY,
        UInt8(1),
    )
    return SparseMatrixComplex{C}(structure, pointer(lu.values))
end

_empty_factorization() = SparseOpaqueFactorizationComplex(
    SPARSE_STATUS_OK,
    SPARSE_ATTRIBUTES_ORDINARY,
    SparseOpaqueSymbolicFactorization(
        SPARSE_STATUS_OK, 0, 0, SPARSE_ATTRIBUTES_ORDINARY, 0, 0,
        Ptr{Cvoid}(C_NULL), 0, 0, 0, 0,
    ),
    false,
    Ptr{Cvoid}(C_NULL),
    0,
    0,
)

"""
    accelerate_sparse_lu(A::SparseMatrixCSC{<:Union{ComplexF32,ComplexF64}}) -> AccelerateSparseLU

Factor `A` with Accelerate's complex sparse LU, in whichever precision `A`
carries. Throws if the factorization does not report `SparseStatusOK`.
"""
function accelerate_sparse_lu(A::SparseMatrixCSC{C,<:Integer}) where {C<:Union{ComplexF32,ComplexF64}}
    size(A, 1) == size(A, 2) ||
        throw(ArgumentError("Accelerate sparse LU requires a square matrix."))
    accelerate_sparse_available() ||
        error("Accelerate sparse LU is unavailable, or its ABI does not match the reference layout.")

    n = size(A, 1)
    # Accelerate wants 0-based CSC with 64-bit column starts and 32-bit row indices.
    column_starts = Vector{Clong}(undef, n + 1)
    @inbounds for index in 1:(n + 1)
        column_starts[index] = Clong(A.colptr[index] - 1)
    end
    row_indices = Vector{Cint}(undef, length(A.rowval))
    @inbounds for index in eachindex(A.rowval)
        row_indices[index] = Cint(A.rowval[index] - 1)
    end

    handle = AccelerateSparseLU{C}(
        _empty_factorization(), column_starts, row_indices, copy(A.nzval), n, false,
    )
    symbolic_options = _default_symbolic_options()
    numeric_options = _default_numeric_options(C)
    factorization = GC.@preserve handle begin
        _raw_factor_lu(_matrix_handle(handle), symbolic_options, numeric_options)
    end
    handle.factorization = factorization
    if factorization.status != SPARSE_STATUS_OK
        handle.freed = true
        error("Accelerate sparse LU failed with status $(factorization.status).")
    end
    finalizer(accelerate_sparse_free!, handle)
    return handle
end

"""
    accelerate_sparse_refactor!(lu, A) -> AccelerateSparseLU

Refactor in place, reusing the existing symbolic analysis. `A` must have exactly
the sparsity pattern the factorization was built from — the caller owns that
check, because Accelerate does not verify it.
"""
function accelerate_sparse_refactor!(
    lu::AccelerateSparseLU{C},
    A::SparseMatrixCSC{C,<:Integer},
) where {C}
    lu.freed && error("Accelerate sparse LU has already been released.")
    length(A.nzval) == length(lu.values) ||
        throw(ArgumentError("Refactorization requires an identical sparsity pattern."))
    copyto!(lu.values, A.nzval)
    numeric_options = _default_numeric_options(C)
    factorization = Ref(lu.factorization)
    workspace = Vector{UInt8}(undef, max(Int(lu.factorization.solve_workspace_required_static), 1))
    GC.@preserve lu workspace begin
        _raw_refactor_lu(_matrix_handle(lu), factorization, numeric_options, pointer(workspace))
    end
    lu.factorization = factorization[]
    lu.factorization.status == SPARSE_STATUS_OK ||
        error("Accelerate sparse refactorization failed with status $(lu.factorization.status).")
    return lu
end

"""
    accelerate_sparse_solve!(X, lu, B) -> X

Solve `A * X = B` for a dense right-hand-side block. `B` and `X` are
column-major matrices with matching dimensions; Accelerate takes the whole block
in one call rather than a column at a time.

Cost grows superlinearly in the number of columns per call, so callers should
feed narrow blocks and parallelise across them rather than passing everything at
once — see `_blocked_accelerate_schur_complement`.
"""
function accelerate_sparse_solve!(
    X::AbstractMatrix{C},
    lu::AccelerateSparseLU{C},
    B::AbstractMatrix{C},
) where {C}
    lu.freed && error("Accelerate sparse LU has already been released.")
    size(B, 1) == lu.size ||
        throw(DimensionMismatch("Right-hand side has $(size(B, 1)) rows, expected $(lu.size)."))
    size(X) == size(B) || throw(DimensionMismatch("Solution and right-hand side shapes differ."))
    X isa StridedMatrix && B isa StridedMatrix ||
        throw(ArgumentError("Accelerate solve requires strided column-major matrices."))

    nrhs = size(B, 2)
    nrhs == 0 && return X
    workspace_size = Int(lu.factorization.solve_workspace_required_static) +
                     nrhs * Int(lu.factorization.solve_workspace_required_per_rhs)
    workspace = Vector{UInt8}(undef, max(workspace_size, 1))
    factorization = Ref(lu.factorization)
    GC.@preserve lu B X workspace begin
        dense_b = DenseMatrixComplex{C}(
            Cint(size(B, 1)), Cint(nrhs), Cint(stride(B, 2)),
            SPARSE_ATTRIBUTES_ORDINARY, pointer(B),
        )
        dense_x = DenseMatrixComplex{C}(
            Cint(size(X, 1)), Cint(nrhs), Cint(stride(X, 2)),
            SPARSE_ATTRIBUTES_ORDINARY, pointer(X),
        )
        _raw_solve(factorization, dense_b, dense_x, pointer(workspace))
    end
    return X
end

function accelerate_sparse_free!(lu::AccelerateSparseLU{C}) where {C}
    lu.freed && return nothing
    lu.freed = true
    factorization = Ref(lu.factorization)
    _raw_destroy(C, factorization)
    return nothing
end

"""
    accelerate_sparse_selftest(; order = 400, rhs = 8) -> NamedTuple

Factor and solve a small random complex system, and report the residual against
a dense reference. A struct-layout mistake shows up here as a wild residual or a
crash, so this runs before the condensation path trusts the binding.
"""
function accelerate_sparse_selftest(
    ::Type{C}=ComplexF32;
    order::Int=400,
    rhs::Int=8,
    seed::Int=1,
) where {C<:Union{ComplexF32,ComplexF64}}
    accelerate_sparse_available() || return (available=false, residual=Inf)
    generator = Random.MersenneTwister(seed)
    A = sprand(generator, C, order, order, 0.01) + C(5) * I
    A = SparseMatrixCSC{C,Int}(A)
    B = rand(generator, C, order, rhs)
    X = similar(B)
    handle = accelerate_sparse_lu(A)
    try
        accelerate_sparse_solve!(X, handle, B)
    finally
        accelerate_sparse_free!(handle)
    end
    residual = norm(A * X - B) / max(norm(B), eps(real(C)))
    return (available=true, residual=residual)
end

end # module
