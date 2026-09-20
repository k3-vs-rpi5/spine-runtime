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
void run_q6_full(spert::Stream& stream, size_t kBlocks, bool signed_a) {
    const size_t kK = 256 * kBlocks;
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
        avec[i] = signed_a ? static_cast<int8_t>((int) (next(seed) % 255) - 127)
                           : static_cast<int8_t>((int) (next(seed) % 127) + 1);

    for (size_t col = 0; col < kCols; col++) {
        d[col] = 1.0f;   /* the operand carries fp16 1.0; the reference must match it */
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
            *reinterpret_cast<uint16_t*>(bb + kQ6Scales + kQ6Qs + kQ6Qh + col * 2) = 0x3c00;
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

    auto future = stream.launch(spert::Grid{1}, [kBlocks](spert::Context* context, const uint8_t* a_ptr,
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
    std::printf("q6full blocks=%zu signed_a=%d worst_abs_diff=%.3f   col0 kernel=%.2f host=%.2f\n",
                kBlocks, (int) signed_a, worst, out[0], host0);
}

/* Which output column consumes which scale lane. Every value is 1 and every scale is
 * 1 except one, so the column that doubles names the pairing. */
void run_q6_scale(spert::Stream& stream, const char* name, size_t scale_lane) {
    constexpr size_t kBlocks = 1;
    std::vector<uint8_t> a(kQ6A * kBlocks, 0);
    std::vector<uint8_t> b(kQ6Bytes * kBlocks, 0);
    std::vector<float>   out(kCols, -1.0f);

    for (size_t blk = 0; blk < kBlocks; blk++) {
        uint8_t* ab = a.data() + blk * kQ6A;

        *reinterpret_cast<float*>(ab)       = 1.0f;
        *reinterpret_cast<int16_t*>(ab + 4) = 0;
        for (size_t i = 0; i < 256; i++)
            reinterpret_cast<int8_t*>(ab + 4 + 2 * 16)[i] = 1;
    }

    for (size_t col = 0; col < kCols; col++) {
        *reinterpret_cast<uint16_t*>(b.data() + kQ6Scales + kQ6Qs + kQ6Qh + col * 2) = 0x3c00;
        for (size_t kii = 0; kii < 16; kii++)
            reinterpret_cast<int8_t*>(b.data())[kii * kCols + col] = (col == scale_lane) ? 2 : 1;
    }

    for (size_t k64 = 0; k64 < 4; k64++) {
        for (size_t sb = 0; sb < 4; sb++) {
            for (size_t b16 = 0; b16 < 16; b16++) {
                for (size_t col = 0; col < kCols; col++) {
                    const uint8_t v = 1;

                    if (sb % 2 == 0) {
                        b[kQ6Scales + ((k64 * 2 + sb / 2) * kCols + col) * 16 + b16] |= (uint8_t) (v & 0x0F);
                    } else {
                        b[kQ6Scales + ((k64 * 2 + sb / 2) * kCols + col) * 16 + b16] |=
                            (uint8_t) ((v & 0x0F) << 4);
                    }
                }
            }
        }
    }

    auto future = stream.launch(spert::Grid{1}, [kBlocks](spert::Context* context, const uint8_t* a_ptr,
                                                          const uint8_t* b_ptr, float* c_ptr) {
        (void) context;
        spacemit_kernels::ime2::gemm_kernel_i8i6k_m1(256, a_ptr, b_ptr, b_ptr, c_ptr, 1, kCols,
                                                     kBlocks, kCols);
    }, a.data(), b.data(), out.data());

    if (!future.valid()) {
        std::printf("RESULT q6scale=%s launch=refused\n", name);
        return;
    }
    future.sync();

    std::printf("q6scale %-18s lane %2zu doubled ->", name, scale_lane);
    for (size_t col = 0; col < kCols; col++)
        if (out[col] < -7936.5f)
            std::printf(" col %zu", col);
    std::printf("   (ref -7936, doubled -15872)\n");
}

/* Which K16 sub-block a scale group belongs to. Values are all 1 and scales all 1
 * except for one sub-block, so the output shifts by -31 x sum(A over that sub-block's
 * slice): 4216 / 12152 / 20088 / 28024 for sub-blocks 0 / 1 / 2 / 3 with A[i] = i+1. */
/* Two non-zero values at once: the smallest case that has values differing across
 * positions, which is what the random comparison says is required to fail. */
void run_q6_pair(spert::Stream& stream, const char* name, size_t k1, uint8_t v1, size_t k2, uint8_t v2) {
    constexpr size_t kBlocks = 1;
    std::vector<uint8_t> a(kQ6A * kBlocks, 0);
    std::vector<uint8_t> b(kQ6Bytes * kBlocks, 0);
    std::vector<float>   out(kCols, -1.0f);
    int8_t               avec[256];

    *reinterpret_cast<float*>(a.data())       = 1.0f;
    *reinterpret_cast<int16_t*>(a.data() + 4) = 0;
    for (size_t i = 0; i < 256; i++) {
        avec[i] = static_cast<int8_t>((i % 100) + 1);
        reinterpret_cast<int8_t*>(a.data() + 4 + 2 * 16)[i] = avec[i];
    }
    for (size_t col = 0; col < kCols; col++) {
        *reinterpret_cast<uint16_t*>(b.data() + kQ6Scales + kQ6Qs + kQ6Qh + col * 2) = 0x3c00;
        for (size_t kii = 0; kii < 16; kii++)
            reinterpret_cast<int8_t*>(b.data())[kii * kCols + col] = 1;
    }
    const size_t where[2] = { k1, k2 };
    const uint8_t vals[2] = { v1, v2 };
    for (int which = 0; which < 2; which++) {
        const size_t kk  = where[which];
        const size_t k64 = kk / 64;
        const size_t sb  = (kk % 64) / 16;
        const size_t b16 = kk % 16;
        const size_t off = kQ6Scales + ((k64 * 2 + sb / 2) * kCols + 0) * 16 + b16;

        if (sb % 2 == 0) {
            b[off] |= (uint8_t) (vals[which] & 0x0F);
        } else {
            b[off] |= (uint8_t) ((vals[which] & 0x0F) << 4);
        }
        b[kQ6Scales + kQ6Qs + (k64 * kCols + 0) * 16 + b16] |=
            (uint8_t) (((vals[which] >> 4) & 3) << (2 * sb));
    }

    auto future = stream.launch(spert::Grid{1}, [kBlocks](spert::Context* context, const uint8_t* a_ptr,
                                                          const uint8_t* b_ptr, float* c_ptr) {
        (void) context;
        spacemit_kernels::ime2::gemm_kernel_i8i6k_m1(256, a_ptr, b_ptr, b_ptr, c_ptr, 1, kCols,
                                                     kBlocks, kCols);
    }, a.data(), b.data(), out.data());
    if (!future.valid()) { std::printf("RESULT q6pair launch=refused\n"); return; }
    future.sync();

    double baseline = 0.0;
    for (size_t i = 0; i < 256; i++) baseline += (double) avec[i] * -32.0;
    const double expected = baseline + (double) v1 * avec[k1] + (double) v2 * avec[k2];
    std::printf("q6pair %-14s kernel_col0=%.1f host=%.1f diff=%.1f\n", name, out[0], expected,
                (double) out[0] - expected);
}

void run_q6_scale_k(spert::Stream& stream, size_t doubled_kii) {
    constexpr size_t kBlocks = 1;
    std::vector<uint8_t> a(kQ6A * kBlocks, 0);
    std::vector<uint8_t> b(kQ6Bytes * kBlocks, 0);
    std::vector<float>   out(kCols, 0.0f);

    for (size_t blk = 0; blk < kBlocks; blk++) {
        uint8_t* ab = a.data() + blk * kQ6A;

        *reinterpret_cast<float*>(ab)       = 1.0f;
        *reinterpret_cast<int16_t*>(ab + 4) = 0;
        for (size_t i = 0; i < 256; i++)
            reinterpret_cast<int8_t*>(ab + 4 + 2 * 16)[i] = static_cast<int8_t>((i % 100) + 1);
    }

    for (size_t col = 0; col < kCols; col++) {
        *reinterpret_cast<uint16_t*>(b.data() + kQ6Scales + kQ6Qs + kQ6Qh + col * 2) = 0x3c00;
        for (size_t kii = 0; kii < 16; kii++)
            reinterpret_cast<int8_t*>(b.data())[kii * kCols + col] = (kii == doubled_kii) ? 2 : 1;
    }

    for (size_t k64 = 0; k64 < 4; k64++) {
        for (size_t sb = 0; sb < 4; sb++) {
            for (size_t b16 = 0; b16 < 16; b16++) {
                for (size_t col = 0; col < kCols; col++) {
                    if (sb % 2 == 0) {
                        b[kQ6Scales + ((k64 * 2 + sb / 2) * kCols + col) * 16 + b16] |= 0x01;
                    } else {
                        b[kQ6Scales + ((k64 * 2 + sb / 2) * kCols + col) * 16 + b16] |= 0x10;
                    }
                }
            }
        }
    }

    auto future = stream.launch(spert::Grid{1}, [kBlocks](spert::Context* context, const uint8_t* a_ptr,
                                                          const uint8_t* b_ptr, float* c_ptr) {
        (void) context;
        spacemit_kernels::ime2::gemm_kernel_i8i6k_m1(256, a_ptr, b_ptr, b_ptr, c_ptr, 1, kCols,
                                                     kBlocks, kCols);
    }, a.data(), b.data(), out.data());

    if (!future.valid()) {
        std::printf("RESULT q6scaleK launch=refused\n");
        return;
    }
    future.sync();

    /* baseline with every scale 1: -31 * sum(A) over 256 values */
    double sum_a = 0.0;
    for (size_t i = 0; i < 256; i++)
        sum_a += (double) ((i % 100) + 1);
    const double baseline = -31.0 * sum_a;
    const double deltas[4] = { -31.0 * 136.0, -31.0 * 392.0, -31.0 * 648.0, -31.0 * 904.0 };
    const double observed = (double) out[0] - baseline;
    int          matched  = -1;

    for (int j = 0; j < 4; j++) {
        if (std::fabs(deltas[j] - observed) < 0.5)
            matched = j;
    }
    std::printf("q6scaleK doubled kii=%2zu  observed_delta=%.1f  matches A slice of sub-block %d"
                "   (expected deltas 4216 12152 20088 28024)\n",
                doubled_kii, observed, matched);
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
    run_q6(stream, "col0 k0 value16", 0, 0, 16);
    run_q6(stream, "col0 k1 value16", 0, 1, 16);
    run_q6(stream, "col0 k0 value32", 0, 0, 32);
    run_q6(stream, "col0 k16 value48", 0, 16, 48);
    run_q6(stream, "col0 k0 value63", 0, 0, 63);
    run_q6(stream, "col0 k16 value1", 0, 16, 1);
    run_q6(stream, "col0 k16 value2", 0, 16, 2);
    run_q6(stream, "col1 k0 value1", 1, 0, 1);
    run_q6(stream, "col0 k32 value1", 0, 32, 1);
    run_q6(stream, "col0 k256 value1", 0, 256, 1);
    run_q6(stream, "col3 k300 value5", 3, 300, 5);
    run_q6_full(stream, 1, false);
    run_q6_full(stream, 1, true);
    for (size_t lane = 0; lane < 4; lane++) {
        char name[32];
        std::snprintf(name, sizeof(name), "lane%zu", lane);
        run_q6_scale(stream, name, lane);
    }
    for (size_t kii = 0; kii < 5; kii++)
        run_q6_scale_k(stream, kii);
    run_q6_pair(stream, "same K16", 0, 5, 1, 7);
    run_q6_pair(stream, "k0+k16", 0, 5, 16, 7);
    run_q6_pair(stream, "k0+k64", 0, 5, 64, 7);
    run_q6_pair(stream, "k0+k7", 0, 5, 7, 7);

    /* Two columns with different values at the same K index: the last untested axis. */
    {
        constexpr size_t kBlocks = 1;
        std::vector<uint8_t> a(kQ6A * kBlocks, 0);
        std::vector<uint8_t> b(kQ6Bytes * kBlocks, 0);
        std::vector<float>   out(kCols, -1.0f);
        int8_t               avec[256];

        *reinterpret_cast<float*>(a.data())       = 1.0f;
        *reinterpret_cast<int16_t*>(a.data() + 4) = 0;
        for (size_t i = 0; i < 256; i++) {
            avec[i] = static_cast<int8_t>((i % 100) + 1);
            reinterpret_cast<int8_t*>(a.data() + 4 + 2 * 16)[i] = avec[i];
        }
        for (size_t col = 0; col < kCols; col++) {
            *reinterpret_cast<uint16_t*>(b.data() + kQ6Scales + kQ6Qs + kQ6Qh + col * 2) = 0x3c00;
            for (size_t kii = 0; kii < 16; kii++)
                reinterpret_cast<int8_t*>(b.data())[kii * kCols + col] = 1;
        }
        /* column 0 value 5 at K0, column 1 value 7 at K0 */
        b[kQ6Scales + 0 * 16 + 0] |= 5;
        b[kQ6Scales + 1 * 16 + 0] |= 7;
        *reinterpret_cast<uint16_t*>(b.data() + kQ6Scales + kQ6Qs + kQ6Qh + 0 * 2) = 0x3c00;

        auto future = stream.launch(spert::Grid{1}, [kBlocks](spert::Context* context,
                                                             const uint8_t* a_ptr, const uint8_t* b_ptr,
                                                             float* c_ptr) {
            (void) context;
            spacemit_kernels::ime2::gemm_kernel_i8i6k_m1(256, a_ptr, b_ptr, b_ptr, c_ptr, 1, kCols,
                                                         kBlocks, kCols);
        }, a.data(), b.data(), out.data());
        future.sync();

        double baseline = 0.0;
        for (size_t i = 0; i < 256; i++) baseline += (double) avec[i] * -32.0;
        const double e0 = baseline + 5.0 * avec[0];
        const double e1 = baseline + 7.0 * avec[0];
        std::printf("q6cols col0 kernel=%.1f host=%.1f   col1 kernel=%.1f host=%.1f\n", out[0], e0,
                    out[1], e1);
    }
    return 0;
}
