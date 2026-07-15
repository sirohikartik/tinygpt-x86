#include<stdexcept>
#include<vector>
#include<cmath>
#include<immintrin.h>

inline float sum_m128(__m128 v) {
    float f[4];
    _mm_storeu_ps(f, v);
    return f[0] + f[1] + f[2] + f[3];
}

inline float max_m128(__m128 v) {
    float f[4];
    _mm_storeu_ps(f, v);
    return std::max({f[0], f[1], f[2], f[3]});
}


class Tensor {
    std::vector<float> data;
    size_t rows;
    size_t cols;

public:

    Tensor() : rows(0), cols(0) {}

    Tensor(std::vector<float> weights, size_t row, size_t col)
        : data(weights), rows(row), cols(col) {}

    size_t shape(int idx) const {
        if(idx==0) return rows;
        return cols;
    }

    const std::vector<float>& getData() const {
        return this->data;
    }

    Tensor operator+(const Tensor& obj) const {
        if (rows != obj.rows || cols != obj.cols)
            throw std::invalid_argument("Shape mismatch for addition");

        std::vector<float> res(data.size());
        size_t n = data.size();

        #pragma omp parallel for if(n > 1024)
        for (size_t i = 0; i < n; i += 4) {
            if (i + 4 <= n) {
                __m128 a = _mm_loadu_ps(&data[i]);
                __m128 b = _mm_loadu_ps(&obj.data[i]);
                _mm_storeu_ps(&res[i], _mm_add_ps(a, b));
            } else {
                for (size_t k = i; k < n; ++k)
                    res[k] = data[k] + obj.data[k];
            }
        }

        return Tensor(res, rows, cols);
    }

    Tensor operator*(const Tensor& obj) const {
        if (cols != obj.rows)
            throw std::invalid_argument("Shape mismatch for matmul: A.cols must equal B.rows");

        const size_t M = rows;
        const size_t K = cols;
        const size_t N = obj.cols;

        std::vector<float> res(M * N, 0.0f);

        // NEON tiled matmul
        constexpr size_t TILE = 32;

        #pragma omp parallel for if(M > 16)
        for (size_t i0 = 0; i0 < M; i0 += TILE) {
            size_t iEnd = std::min(i0 + TILE, M);
            for (size_t k0 = 0; k0 < K; k0 += TILE) {
                size_t kEnd = std::min(k0 + TILE, K);
                for (size_t j0 = 0; j0 < N; j0 += TILE) {
                    size_t jEnd = std::min(j0 + TILE, N);
                    for (size_t i = i0; i < iEnd; ++i) {
                        for (size_t k = k0; k < kEnd; ++k) {
                            __m128 a_ik = _mm_set1_ps(data[i * K + k]);
                            size_t j = j0;
                            // process 4 columns at a time
                            for (; j + 4 <= jEnd; j += 4) {
                                __m128 b = _mm_loadu_ps(&obj.data[k * N + j]);
                                __m128 c = _mm_loadu_ps(&res[i * N + j]);
                                c = _mm_add_ps(c, _mm_mul_ps(a_ik, b));  // c += a * b
                                _mm_storeu_ps(&res[i * N + j], c);
                            }
                            // remainder
                            for (; j < jEnd; ++j)
                                res[i * N + j] += data[i * K + k] * obj.data[k * N + j];
                        }
                    }
                }
            }
        }

        return Tensor(res, M, N);
    }

    Tensor LayerNorm(const Tensor& gamma, const Tensor& beta, float E = 1e-5f) const {
        std::vector<float> res(data.size());

        for(size_t i = 0; i < rows; i++){
            const float* row = &data[i * cols];

            // compute mean with SSE
            __m128 sum_vec = _mm_setzero_ps();
            size_t j = 0;
            for (; j + 4 <= cols; j += 4)
                sum_vec = _mm_add_ps(sum_vec, _mm_loadu_ps(row + j));
            float mean = sum_m128(sum_vec);
            for (; j < cols; j++) mean += row[j];
            mean /= cols;

            // compute variance with SSE
            __m128 var_vec = _mm_setzero_ps();
            __m128 mean_vec = _mm_set1_ps(mean);
            j = 0;
            for (; j + 4 <= cols; j += 4) {
                __m128 diff = _mm_sub_ps(_mm_loadu_ps(row + j), mean_vec);
                var_vec = _mm_add_ps(var_vec, _mm_mul_ps(diff, diff));
            }
            float var = sum_m128(var_vec);
            for (; j < cols; j++) {
                float diff = row[j] - mean;
                var += diff * diff;
            }
            var /= cols;

            float inv_std = 1.0f / std::sqrt(var + E);
            __m128 inv_std_vec = _mm_set1_ps(inv_std);
            float* out_row = &res[i * cols];

            j = 0;
            for (; j + 4 <= cols; j += 4) {
                __m128 norm = _mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(row + j), mean_vec), inv_std_vec);
                __m128 g = _mm_loadu_ps(&gamma.data[j]);
                __m128 b = _mm_loadu_ps(&beta.data[j]);
                _mm_storeu_ps(out_row + j, _mm_add_ps(b, _mm_mul_ps(norm, g)));
            }
            for (; j < cols; j++) {
                float norm = (row[j] - mean) * inv_std;
                out_row[j] = gamma.data[j] * norm + beta.data[j];
            }
        }
        return Tensor(res, rows, cols);
    }

    Tensor operator/(const float val) const {
        std::vector<float> res(rows * cols);
        __m128 val_vec = _mm_set1_ps(val);
        size_t i = 0;
        for (; i + 4 <= data.size(); i += 4)
            _mm_storeu_ps(&res[i], _mm_div_ps(_mm_loadu_ps(&data[i]), val_vec));
        for (; i < data.size(); i++)
            res[i] = data[i] / val;
        return Tensor(res, rows, cols);
    }

    Tensor softmax() const {
        std::vector<float> res(data.size());

        #pragma omp parallel for if(rows > 8)
        for(size_t i = 0; i < rows; i++) {
            const float* row = &data[i * cols];
            float* out = &res[i * cols];

            // find max
            __m128 max_vec = _mm_set1_ps(row[0]);
            size_t j = 0;
            for (; j + 4 <= cols; j += 4)
                max_vec = _mm_max_ps(max_vec, _mm_loadu_ps(row + j));
            float max_val = max_m128(max_vec);
            for (; j < cols; j++) max_val = std::max(max_val, row[j]);

            // exp and sum
            __m128 max_bcast = _mm_set1_ps(max_val);
            float sum_exp = 0.0f;
            j = 0;
            for (; j + 4 <= cols; j += 4) {
                __m128 v = _mm_sub_ps(_mm_loadu_ps(row + j), max_bcast);
                // no SSE exp - compute scalar
                float tmp[4];
                _mm_storeu_ps(tmp, v);
                tmp[0] = std::exp(tmp[0]);
                tmp[1] = std::exp(tmp[1]);
                tmp[2] = std::exp(tmp[2]);
                tmp[3] = std::exp(tmp[3]);
                __m128 ev = _mm_loadu_ps(tmp);
                _mm_storeu_ps(out + j, ev);
                sum_exp += tmp[0] + tmp[1] + tmp[2] + tmp[3];
            }
            for (; j < cols; j++) {
                out[j] = std::exp(row[j] - max_val);
                sum_exp += out[j];
            }

            // normalize
            __m128 inv_sum = _mm_set1_ps(1.0f / sum_exp);
            j = 0;
            for (; j + 4 <= cols; j += 4)
                _mm_storeu_ps(out + j, _mm_mul_ps(_mm_loadu_ps(out + j), inv_sum));
            for (; j < cols; j++) out[j] /= sum_exp;
        }

        return Tensor(res, rows, cols);
    }

    Tensor t() const {
        std::vector<float> res(data.size());
        for(size_t i = 0; i < rows; i++)
            for(size_t j = 0; j < cols; j++)
                res[j * rows + i] = data[i * cols + j];
        return Tensor(res, cols, rows);
    }

    Tensor mask() const {
        std::vector<float> res = data;
        const float MASK_VALUE = -1e9f;
        for(size_t i = 0; i < rows; i++)
            for(size_t j = i + 1; j < cols; j++)
                res[i * cols + j] = MASK_VALUE;
        return Tensor(res, rows, cols);
    }

    static Tensor concat_horizontal(const std::vector<Tensor>& tensors) {
        if (tensors.empty()) return Tensor();

        size_t rows = tensors[0].rows;
        size_t total_cols = 0;
        for (const auto& t : tensors) {
            if (t.rows != rows)
                throw std::invalid_argument("All tensors must have same number of rows for horizontal concat");
            total_cols += t.cols;
        }

        std::vector<float> res(rows * total_cols);
        size_t col_offset = 0;
        for (const auto& t : tensors) {
            for (size_t i = 0; i < rows; i++)
                std::copy(t.data.begin() + i * t.cols,
                          t.data.begin() + i * t.cols + t.cols,
                          res.begin() + i * total_cols + col_offset);
            col_offset += t.cols;
        }
        return Tensor(res, rows, total_cols);
    }

    static Tensor concat_vertical(const std::vector<Tensor>& tensors) {
        if (tensors.empty()) return Tensor();

        size_t cols = tensors[0].cols;
        size_t total_rows = 0;
        for (const auto& t : tensors) {
            if (t.cols != cols)
                throw std::invalid_argument("All tensors must have same number of columns for vertical concat");
            total_rows += t.rows;
        }

        std::vector<float> res(total_rows * cols);
        size_t row_offset = 0;
        for (const auto& t : tensors) {
            std::copy(t.data.begin(), t.data.end(), res.begin() + row_offset * cols);
            row_offset += t.rows;
        }
        return Tensor(res, total_rows, cols);
    }

    Tensor add_bias(const Tensor& bias) const {
        std::vector<float> res(data.size());
        #pragma omp parallel for if(rows > 8)
        for(size_t i = 0; i < rows; i++) {
            size_t j = 0;
            for (; j + 4 <= cols; j += 4) {
                __m128 a = _mm_loadu_ps(&data[i * cols + j]);
                __m128 b = _mm_loadu_ps(&bias.data[j]);
                _mm_storeu_ps(&res[i * cols + j], _mm_add_ps(a, b));
            }
            for (; j < cols; j++)
                res[i * cols + j] = data[i * cols + j] + bias.data[j];
        }
        return Tensor(res, rows, cols);
    }
};
