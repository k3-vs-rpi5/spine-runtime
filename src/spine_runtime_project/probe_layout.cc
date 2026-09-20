// SPDX-License-Identifier: Apache-2.0
/*
 * Which A value pairs with which B nibble, and which output column it lands in.
 *
 * Why: a new kernel for a packed six-bit operand has to know how `vmadot` pairs a
 * materialised byte with the A slice. The instruction set is not documented in this
 * workspace, so the pairing is measured instead of assumed - by driving the vendor's
 * own `gemm_kernel_i8i4_m1`, whose operand layout its reference implementation states,
 * with a single non-zero nibble and a strictly increasing A slice. The output column
 * and the value it holds then name the pairing directly.
 *
 * The kernel must execute on an AI core, so the call is wrapped in a one-core
 * `spert::Stream` launch even though the work is trivially small.
 */
#include "spert.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

namespace spacemit_kernels {
namespace ime2 {
size_t gemm_kernel_i8i4_m1(size_t          blk_len,
                           const uint8_t * quant_a_ptr,
                           const uint8_t * quant_b_data,
                           const uint8_t * quant_b_zp,
                           float *         c_ptr,
                           size_t          count_m,
                           size_t          count_n,
                           size_t          k_blks,
                           size_t          ldc);
/* This one the project's own patch defines with a void return, so the declaration
 * has to match its mangled name. */
void gemm_kernel_i8i6k_m1(size_t          blk_len,
                          const uint8_t * quant_a_ptr,
                          const uint8_t * quant_b_data,
                          const uint8_t * quant_b_zp,
                          float *         c_ptr,
                          size_t          count_m,
                          size_t          count_n,
                          size_t          k_blks,
                          size_t          ldc);
}
}  // namespace spacemit_kernels

namespace {

constexpr size_t kBlkLen = 32;
constexpr size_t kCols   = 32;
/* A: [fp32 scale][int16 sum][32 int8]; B: [32 fp16 d][32 u8 zp][32 * 16 nibbles]. */
constexpr size_t kABytes = 4 + 2 + kBlkLen;
constexpr size_t kBBytes = 32 * 2 + 32 + kCols * (kBlkLen / 2);

/* The compact six-bit block, 32 columns by 256 values: scales[512] | qs[4096] |
 * qh[2048] | fp16 d[32]. */
constexpr size_t kQ6Scales = 32 * 16;
constexpr size_t kQ6Qs     = 32 * 128;
constexpr size_t kQ6Qh     = 32 * 64;
constexpr size_t kQ6Bytes  = kQ6Scales + kQ6Qs + kQ6Qh + 32 * 2;
constexpr size_t kQ6A      = 4 + 2 * 16 + 256;

struct case_t {
    const char* name;
    size_t      column;
    size_t      byte;
    bool        high_nibble;
};

void run_one(spert::Stream& stream, const case_t& c) {
    std::vector<uint8_t> a(kABytes, 0);
    std::vector<uint8_t> b(kBBytes, 0);
    std::vector<float>   out(kCols, -1.0f);

    *reinterpret_cast<float*>(a.data())        = 1.0f;             /* a_scale */
    *reinterpret_cast<int16_t*>(a.data() + 4)  = 0;                /* a_sum */
    for (size_t i = 0; i < kBlkLen; i++)                            /* A[i] = i + 1 */
        reinterpret_cast<int8_t*>(a.data() + 6)[i] = static_cast<int8_t>(i + 1);

    for (size_t col = 0; col < kCols; col++)
        reinterpret_cast<uint16_t*>(b.data())[col] = 0x3c00;        /* fp16 1.0 */
    /* the zero points stay 0; the nibble under test lives at column c.column */
    uint8_t* nibbles = b.data() + 32 * 2 + 32;
    nibbles[c.column * (kBlkLen / 2) + c.byte] =
        static_cast<uint8_t>(c.high_nibble ? 0x10 : 0x01);

    auto future = stream.launch(spert::Grid{1}, [](spert::Context* context, const uint8_t* a_ptr,
                                                   const uint8_t* b_ptr, float* c_ptr) {
        (void) context;
        spacemit_kernels::ime2::gemm_kernel_i8i4_m1(kBlkLen, a_ptr, b_ptr, b_ptr, c_ptr, 1, kCols, 1,
                                                    kCols);
    }, a.data(), b.data(), out.data());

    if (!future.valid()) {
        std::printf("RESULT case=%s launch=refused\n", c.name);
        return;
    }
    future.sync();

    std::printf("case %-22s ", c.name);
    for (size_t col = 0; col < kCols; col++) {
        if (out[col] != 0.0f)
            std::printf("col %2zu -> %.1f   ", col, out[col]);
    }
    std::printf("\n");
}

/* One non-zero value in the compact operand at (column, K index), A[i] = i + 1, so
 * the output column and the number it holds name both halves of the pairing. */
void run_q6(spert::Stream& stream, const char* name, size_t column, size_t k_index, uint8_t value) {
    constexpr size_t kBlocks = 2;   /* two K256 superblocks, so the pointer walk is exercised */
    std::vector<uint8_t> a(kQ6A * kBlocks, 0);
    std::vector<uint8_t> b(kQ6Bytes * kBlocks, 0);
    std::vector<float>   out(kCols, -1.0f);

    for (size_t blk = 0; blk < kBlocks; blk++) {
        uint8_t* ab = a.data() + blk * kQ6A;

        *reinterpret_cast<float*>(ab)       = 1.0f;
        *reinterpret_cast<int16_t*>(ab + 4) = 0;
        for (size_t i = 0; i < 256; i++)
            reinterpret_cast<int8_t*>(ab + 4 + 2 * 16)[i] = static_cast<int8_t>((i % 100) + 1);
    }

    for (size_t col = 0; col < kCols; col++) {
        for (size_t blk = 0; blk < kBlocks; blk++) {
            uint8_t* bb = b.data() + blk * kQ6Bytes;

            *reinterpret_cast<uint16_t*>(bb + kQ6Scales + kQ6Qs + kQ6Qh + col * 2) = 0x3c00;
            for (size_t kii = 0; kii < 16; kii++)
                reinterpret_cast<int8_t*>(bb)[kii * 32 + col] = 1;
        }
    }

    const size_t blk = k_index / 256;
    const size_t kk  = k_index % 256;
    const size_t k64 = kk / 64;
    const size_t sb  = (kk % 64) / 16;
    const size_t b16 = kk % 16;
    uint8_t*     bb  = b.data() + blk * kQ6Bytes;

    if (sb % 2 == 0) {
        bb[kQ6Scales + ((k64 * 2 + sb / 2) * kCols + column) * 16 + b16] |= (uint8_t) (value & 0x0F);
    } else {
        bb[kQ6Scales + ((k64 * 2 + sb / 2) * kCols + column) * 16 + b16] |=
            (uint8_t) ((value & 0x0F) << 4);
    }
    bb[kQ6Scales + kQ6Qs + (k64 * kCols + column) * 16 + b16] |=
        (uint8_t) (((value >> 4) & 3) << (2 * sb));

    auto future = stream.launch(spert::Grid{1}, [](spert::Context* context, const uint8_t* a_ptr,
                                                   const uint8_t* b_ptr, float* c_ptr) {
        (void) context;
        spacemit_kernels::ime2::gemm_kernel_i8i6k_m1(256, a_ptr, b_ptr, b_ptr, c_ptr, 1, kCols, 2,
                                                     kCols);
    }, a.data(), b.data(), out.data());

    if (!future.valid()) {
        std::printf("RESULT q6=%s launch=refused\n", name);
        return;
    }
    future.sync();

    std::printf("q6 %-22s ", name);
    for (size_t col = 0; col < kCols; col++)
        if (out[col] != 0.0f)
            std::printf("col %2zu -> %.1f   ", col, out[col]);
    std::printf("\n");
}

/* The whole thing at the real size: K = 2048 (eight superblocks), random values and
 * random activations, against a host-side reference of the same operand bytes. A
 * degenerate single-value case cannot catch a plane or pointer mistake; this can. */
void run_q6_full(spert::Stream& stream) {
    constexpr size_t kBlocks = 8;
    constexpr size_t kK      = 256 * kBlocks;
    std::vector<uint8_t> a(kQ6A * kBlocks, 0);
    std::vector<uint8_t> b(kQ6Bytes * kBlocks, 0);
    std::vector<float>   out(kCols, 0.0f);
    std::vector<uint8_t> values(kCols * kK, 0);
    std::vector<int8_t>  avec(kK, 0);
    std::vector<int8_t>  scales(kCols * (kK / 16), 1);
    std::vector<float>   d(kCols, 1.0f);

    auto next = [](unsigned& seed) { seed = seed * 1103515245u + 12345u; return (seed >> 16) & 0x7fff; };
    unsigned seed = 12345u;

    for (size_t i = 0; i < kK; i++)
        avec[i] = static_cast<int8_t>((int) (next(seed) % 255) - 127);

    for (size_t col = 0; col < kCols; col++) {
        d[col] = static_cast<float>((int) (next(seed) % 100) + 1) / 100.0f;
        for (size_t kii = 0; kii < kK / 16; kii++)
            scales[col * (kK / 16) + kii] = static_cast<int8_t>((int) (next(seed) % 31) - 15);
        for (size_t k = 0; k < kK; k++)
            values[col * kK + k] = static_cast<uint8_t>(next(seed) & 0x3F);
    }

    for (size_t blk = 0; blk < kBlocks; blk++) {
        uint8_t* ab = a.data() + blk * kQ6A;

        *reinterpret_cast<float*>(ab)       = 1.0f;
        *reinterpret_cast<int16_t*>(ab + 4) = 0;
        for (size_t i = 0; i < 256; i++)
            reinterpret_cast<int8_t*>(ab + 4 + 2 * 16)[i] = avec[blk * 256 + i];
    }

    for (size_t blk = 0; blk < kBlocks; blk++) {
        uint8_t* bb = b.data() + blk * kQ6Bytes;

        for (size_t col = 0; col < kCols; col++) {
            const size_t base = blk * 256;

            *reinterpret_cast<uint16_t*>(bb + kQ6Scales + kQ6Qs + kQ6Qh + col * 2) =
                (uint16_t) (0x3c00 | (col & 1));   /* fp16 near 1.0, column-dependent */
            for (size_t kii = 0; kii < 16; kii++)
                reinterpret_cast<int8_t*>(bb)[kii * kCols + col] =
                    scales[col * (kK / 16) + blk * 16 + kii];
        }

        for (size_t col = 0; col < kCols; col++) {
            for (size_t k64 = 0; k64 < 4; k64++) {
                for (size_t sb = 0; sb < 4; sb++) {
                    for (size_t b16 = 0; b16 < 16; b16++) {
                        const size_t  j = blk * 256 + k64 * 64 + sb * 16 + b16;
                        const uint8_t v = values[col * kK + j];

                        if (sb % 2 == 0) {
                            bb[kQ6Scales + ((k64 * 2 + sb / 2) * kCols + col) * 16 + b16] |=
                                (uint8_t) (v & 0x0F);
                        } else {
                            bb[kQ6Scales + ((k64 * 2 + sb / 2) * kCols + col) * 16 + b16] |=
                                (uint8_t) ((v & 0x0F) << 4);
                        }
                        bb[kQ6Scales + kQ6Qs + (k64 * kCols + col) * 16 + b16] |=
                            (uint8_t) (((v >> 4) & 3) << (2 * sb));
                    }
                }
            }
        }
    }

    auto future = stream.launch(spert::Grid{1}, [](spert::Context* context, const uint8_t* a_ptr,
                                                   const uint8_t* b_ptr, float* c_ptr) {
        (void) context;
        spacemit_kernels::ime2::gemm_kernel_i8i6k_m1(256, a_ptr, b_ptr, b_ptr, c_ptr, 1, kCols,
                                                     kBlocks, kCols);
    }, a.data(), b.data(), out.data());

    if (!future.valid()) {
        std::printf("RESULT q6full launch=refused\n");
        return;
    }
    future.sync();

    /* host reference over the same bytes */
    float worst = 0.0f;
    float host0 = 0.0f;
    for (size_t col = 0; col < kCols; col++) {
        double acc = 0.0;
        for (size_t k = 0; k < kK; k++) {
            const double s = (double) scales[col * (kK / 16) + k / 16];
            acc += (double) avec[k] * (double) ((int) values[col * kK + k] - 32) * s;
        }
        const double expected = acc * (double) d[col];
        const double diff     = std::fabs(expected - (double) out[col]);
        if (col == 0)
            host0 = (float) expected;
        if (diff > worst)
            worst = (float) diff;
    }
    std::printf("q6full worst_abs_diff=%.3f   col0 kernel=%.2f host=%.2f\n", worst, out[0], host0);
}

}  // namespace

int main() {
    spert::BackendHandle k3 = spert::backend("spacemit-k3");

    if (!k3.valid()) {
        std::printf("RESULT probe=skipped-reason=a100-backend-unavailable\n");
        return 3;
    }

    spert::StreamConfig config;
    config.n_cores  = 1;
    config.core_ids = { 8 };
    spert::Stream stream(k3, config);

    if (!stream.valid()) {
        std::printf("RESULT probe=skipped-reason=stream-refused\n");
        return 4;
    }

    /* A[i] = i+1, so a single non-zero nibble prints the A value it paired with. */
    const case_t cases[] = {
        { "col0 byte0 low", 0, 0, false },
        { "col0 byte0 high", 0, 0, true },
        { "col0 byte1 low", 0, 1, false },
        { "col1 byte0 low", 1, 0, false },
        { "col8 byte0 low", 8, 0, false },
        { "col0 byte8 low", 0, 8, false },
    };

    for (const case_t& c : cases)
        run_one(stream, c);
    run_q6(stream, "col0 k0 value1", 0, 0, 1);
    run_q6(stream, "col0 k1 value1", 0, 1, 1);
    run_q6(stream, "col0 k16 value1", 0, 16, 1);
    run_q6(stream, "col0 k16 value2", 0, 16, 2);
    run_q6(stream, "col1 k0 value1", 1, 0, 1);
    run_q6(stream, "col0 k32 value1", 0, 32, 1);
    run_q6(stream, "col0 k256 value1", 0, 256, 1);
    run_q6(stream, "col3 k300 value5", 3, 300, 5);
    run_q6_full(stream);
    return 0;
}
