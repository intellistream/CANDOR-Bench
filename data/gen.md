#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>

using namespace std;
using Vec = vector<float>;

// 将一组向量写入 .fvecs 文件
// 格式：每条记录先写入 int dim（维度），再写入 dim 个 float
void write_fvecs(const string &filename, const vector<Vec> &data) {
    ofstream out(filename, ios::binary);
    if (!out) {
        cerr << "Cannot open " << filename << " for writing.\n";
        return;
    }
    int dim = data.empty() ? 0 : data[0].size();
    for (const auto &v : data) {
        out.write(reinterpret_cast<const char *>(&dim), sizeof(int));
        out.write(reinterpret_cast<const char *>(v.data()), dim * sizeof(float));
    }
    out.close();
}

void gen_G(int D, int N_learn, int N_query, vector<Vec> &learn, vector<Vec> &query) {
    /*
     * 数据集分布：全局基线分布
     * - Learn集：标准正态分布 N(0,1)，所有维度都遵循相同的分布
     * - Query集：标准正态分布 N(0,1)，与Learn集完全相同的分布
     * 
     * 特点：无分布偏移，用于作为基准测试，验证算法在稳定分布下的性能
     */
    default_random_engine eng(42);
    normal_distribution<float> gauss(0.0f, 1.0f);

    // Learn 集：全局基线分布，标准正态
    learn.reserve(N_learn);
    for (int i = 0; i < N_learn; ++i) {
        Vec v(D);
        for (int d = 0; d < D; ++d) v[d] = gauss(eng);
        learn.push_back(v);
    }

    //// Query 集：第 0 维缓慢线性平移，其他维保持标准正态
    //float shift_max = 5.0f;
    query.reserve(N_query);
    for (int i = 0; i < N_query; ++i) {
        //float mu = shift_max * i / float(N_query - 1);
        Vec v(D);
        for (int d = 0; d < D; ++d) {
            //v[d] = (d == 0 ? gauss(eng) + mu : gauss(eng));
            v[d] = gauss(eng);
        }
        query.push_back(v);
    }
}

// 生成 Pattern A1：方向性轻微转移
// 理由：模拟数据分布沿单一维度（第 0 维）缓慢移动，体现长期趋势。
// learn：标准正态分布；query：在第 0 维上逐渐平滑地添加从 0 到 shift_max 的偏移。
void gen_A1(int D, int N_learn, int N_query, vector<Vec> &learn, vector<Vec> &query) {
    /*
     * 数据集分布：方向性轻微转移
     * - Learn集：标准正态分布 N(0,1)，所有维度都遵循相同的分布
     * - Query集：在第0维上存在线性偏移，偏移量从0逐渐增加到5.0
     *           其他维度保持标准正态分布 N(0,1)
     * 
     * 特点：模拟数据分布沿单一维度缓慢移动的长期趋势，偏移是平滑和可预测的
     */
    default_random_engine eng(42);
    normal_distribution<float> gauss(0.0f, 1.0f);

    // Learn 集：全局基线分布，标准正态
    learn.reserve(N_learn);
    for (int i = 0; i < N_learn; ++i) {
        Vec v(D);
        for (int d = 0; d < D; ++d) v[d] = gauss(eng);
        learn.push_back(v);
    }

    // Query 集：第 0 维缓慢线性平移，其他维保持标准正态
    float shift_max = 5.0f;
    query.reserve(N_query);
    for (int i = 0; i < N_query; ++i) {
        float mu = shift_max * i / float(N_query - 1);
        Vec v(D);
        for (int d = 0; d < D; ++d) {
            v[d] = (d == 0 ? gauss(eng) + mu : gauss(eng));
        }
        query.push_back(v);
    }
}

// 生成 Pattern A2：局部轻微转移
// 理由：模拟数据分布在小范围内波动，但不产生全新模式。
// learn：N(0, 0.5)；query：N(0, 0.7)，方差略增，体现轻微但局部的漂移。
void gen_A2(int D, int N_learn, int N_query, vector<Vec> &learn, vector<Vec> &query) {
    /*
     * 数据集分布：局部轻微转移
     * - Learn集：正态分布 N(0, 0.5)，方差较小，数据聚集在原点附近
     * - Query集：正态分布 N(0, 0.7)，方差略大于Learn集，体现轻微的分布漂移
     * 
     * 特点：模拟数据分布在小范围内的波动，方差增加但中心位置不变，
     *       体现轻微但局部的分布变化，不产生全新的模式
     */
    default_random_engine eng(43);
    normal_distribution<float> gauss_learn(0.0f, 0.5f);
    normal_distribution<float> gauss_query(0.0f, 0.7f);

    // Learn 和 query 都聚集在原点附近，但 query 方差稍大
    learn.reserve(N_learn);
    for (int i = 0; i < N_learn; ++i) {
        Vec v(D);
        for (int d = 0; d < D; ++d) v[d] = gauss_learn(eng);
        learn.push_back(v);
    }
    query.reserve(N_query);
    for (int i = 0; i < N_query; ++i) {
        Vec v(D);
        for (int d = 0; d < D; ++d) v[d] = gauss_query(eng);
        query.push_back(v);
    }
}

// 生成 Pattern B：突发性严重转移
// 理由：模拟学习集每隔 5000 条发生一次大跳跃，query 全部位于最后一个分布。
// 这样在流式场景下，每隔固定条目触发一次显著分布突变。
//\ n// learn: 划分为 N_learn/5000 段，每段中心沿着固定方向大幅偏移。
// query: 全部集中在最后一个中心附近。
void gen_B1(int D, int N_learn, int N_query, vector<Vec> &learn, vector<Vec> &query) {
    /*
     * 数据集分布：突发性严重转移（固定方向）
     * - Learn集：划分为多个段，每段5000个样本
     *           每段的中心沿固定方向大幅偏移（偏移量10.0）
     *           每个段内的样本在中心附近散布（N(center, 3.0)）
     * - Query集：全部集中在最后一个段的中心附近（N(last_center, 1.0)）
     * 
     * 特点：模拟数据分布发生突发性大跳跃，方向固定但幅度很大，
     *       体现流式场景下的显著分布突变
     */
    default_random_engine eng(44);
    normal_distribution<float> gauss(0.0f, 1.0f);
    normal_distribution<float> gauss_tmp(0.0f, 3.0f);

    // 定义第一个中心，后续每段在该基础上添加较大偏移delta
    Vec base(D);
    for (int d = 0; d < D; ++d) base[d] = gauss(eng);
    Vec delta(D);
    float shift_size = 10.0f;  // 每次突变幅度
    for (int d = 0; d < D; ++d) delta[d] = (gauss(eng) >= 0 ? 1 : -1) * shift_size;

    int segment_size = 5000;
    int num_segments = (N_learn + segment_size - 1) / segment_size;
    learn.reserve(N_learn);
    for (int i = 0; i < N_learn; ++i) {
        int seg = i / segment_size;
        Vec center(D);
        for (int d = 0; d < D; ++d)
            // 每个段的中心按 seg * delta 移动
            center[d] = base[d] + seg * delta[d];
        Vec v(D);
        for (int d = 0; d < D; ++d)
            v[d] = center[d] + gauss_tmp(eng);
        learn.push_back(v);
    }

    // Query：全部在最后一个段中心附近，模拟一次大的全局突变后的新分布
    Vec last_center(D);
    for (int d = 0; d < D; ++d)
        last_center[d] = base[d] + (num_segments - 1) * delta[d];
    query.reserve(N_query);
    for (int i = 0; i < N_query; ++i) {
        Vec v(D);
        for (int d = 0; d < D; ++d)
            v[d] = last_center[d] + gauss(eng);
        query.push_back(v);
    }
}

void gen_B2(int D, int N_learn, int N_query, vector<Vec> &learn, vector<Vec> &query) {
    /*
     * 数据集分布：突发性严重转移（随机方向）
     * - Learn集：划分为多个段，每段5000个样本
     *           每段的中心沿随机方向大幅偏移（偏移量10.0）
     *           每个段内的样本在中心附近散布（N(center, 3.0)）
     * - Query集：全部集中在最后一个段的中心附近（N(last_center, 1.0)）
     * 
     * 特点：与B1类似，但偏移方向是随机的，更真实地模拟不可预测的分布突变
     */
    default_random_engine eng(47); // 新的随机种子
    normal_distribution<float> gauss_center_components(0.0f, 1.0f); // 用于生成中心点、偏移方向分量、查询点
    normal_distribution<float> gauss_learn_spread(0.0f, 3.0f);    // 用于生成学习点在中心点周围的散布

    Vec current_segment_center(D);
    // 初始化第一个段的中心
    for (int d = 0; d < D; ++d) {
        current_segment_center[d] = gauss_center_components(eng);
    }

    float shift_magnitude = 10.0f;  // 每次突变的幅度
    int segment_size = 5000;        // 每个段的大小

    learn.reserve(N_learn);
    for (int i = 0; i < N_learn; ++i) {
        // 当达到新段的起点时 (除了第一个点)，进行一次随机方向的突变
        if (i > 0 && i % segment_size == 0) {
            Vec random_shift_direction(D);
            float norm_sq = 0.0f;
            for (int d = 0; d < D; ++d) {
                random_shift_direction[d] = gauss_center_components(eng);
                norm_sq += random_shift_direction[d] * random_shift_direction[d];
            }
            float norm = std::sqrt(norm_sq);
            if (norm == 0) norm = 1.0f; // 防止除以零

            // 更新当前段的中心
            for (int d = 0; d < D; ++d) {
                current_segment_center[d] += (random_shift_direction[d] / norm) * shift_magnitude;
            }
        }

        // 在当前段的中心附近生成学习点
        Vec v(D);
        for (int d = 0; d < D; ++d) {
            v[d] = current_segment_center[d] + gauss_learn_spread(eng);
        }
        learn.push_back(v);
    }

    // Query 集：全部在学习集最后一个段的中心附近生成
    query.reserve(N_query);
    for (int i = 0; i < N_query; ++i) {
        Vec v(D);
        for (int d = 0; d < D; ++d) {
            v[d] = current_segment_center[d] + gauss_center_components(eng); // 查询点用标准差为1的高斯分布
        }
        query.push_back(v);
    }
}

// 生成 Pattern B_rand_last：突发性严重转移（随机方向），Query 在最后一段
// 理由：模拟学习集每隔一段发生一次大跳跃（方向随机），Query 集中在学习集演变的最终状态附近。
// learn: 划分为若干段，每段中心在前一段中心的基础上沿随机方向大幅偏移。
// query: 全部集中在学习集最后一个段的中心附近。
void gen_B3(int D, int N_learn, int N_query, vector<Vec> &learn, vector<Vec> &query) {
    /*
     * 数据集分布：突发性严重转移（随机方向，大段）
     * - Learn集：划分为多个段，每段30000个样本
     *           每段的中心沿随机方向大幅偏移（偏移量10.0）
     *           每个段内的样本在中心附近散布（N(center, 3.0)）
     * - Query集：全部集中在最后一个段的中心附近（N(last_center, 1.0)）
     * 
     * 特点：与B2类似，但段大小更大（30000 vs 5000），
     *       模拟更长时间间隔的分布突变，Query集中在最终状态
     */
    default_random_engine eng(47); // 保持与原 gen_B_rand 一致的种子，确保 learn 集相同
    normal_distribution<float> gauss_center_components(0.0f, 1.0f);
    normal_distribution<float> gauss_learn_spread(0.0f, 3.0f);

    Vec current_segment_center(D);
    for (int d = 0; d < D; ++d) {
        current_segment_center[d] = gauss_center_components(eng);
    }

    float shift_magnitude = 10.0f;
    int segment_size = 30000;

    learn.reserve(N_learn);
    for (int i = 0; i < N_learn; ++i) {
        if (i > 0 && i % segment_size == 0) {
            Vec random_shift_direction(D);
            float norm_sq = 0.0f;
            for (int d = 0; d < D; ++d) {
                random_shift_direction[d] = gauss_center_components(eng);
                norm_sq += random_shift_direction[d] * random_shift_direction[d];
            }
            float norm = std::sqrt(norm_sq);
            if (norm == 0) norm = 1.0f;

            for (int d = 0; d < D; ++d) {
                current_segment_center[d] += (random_shift_direction[d] / norm) * shift_magnitude;
            }
        }
        Vec v(D);
        for (int d = 0; d < D; ++d) {
            v[d] = current_segment_center[d] + gauss_learn_spread(eng);
        }
        learn.push_back(v);
    }

    query.reserve(N_query);
    for (int i = 0; i < N_query; ++i) {
        Vec v(D);
        for (int d = 0; d < D; ++d) {
            v[d] = current_segment_center[d] + gauss_center_components(eng);
        }
        query.push_back(v);
    }
}

// 生成 Pattern B_rand_first：突发性严重转移（随机方向），Query 在第一段
// 理由：模拟学习集每隔一段发生一次大跳跃（方向随机），Query 集中在学习集演变的初始状态附近。
// learn: 与 gen_B_rand_last 生成方式相同。
// query: 全部集中在学习集第一个段的中心附近。
void gen_B4(int D, int N_learn, int N_query, vector<Vec> &learn, vector<Vec> &query) {
    /*
     * 数据集分布：突发性严重转移（随机方向，大段）
     * - Learn集：与B3完全相同，划分为多个段，每段30000个样本
     *           每段的中心沿随机方向大幅偏移（偏移量10.0）
     *           每个段内的样本在中心附近散布（N(center, 3.0)）
     * - Query集：全部集中在第一个段的中心附近（N(first_center, 1.0)）
     * 
     * 特点：与B3的Learn集完全相同，但Query集中在初始状态，
     *       用于测试算法对历史模式的记忆能力
     */
    default_random_engine eng(47); // 保持与原 gen_B_rand 一致的种子，确保 learn 集相同
    normal_distribution<float> gauss_center_components(0.0f, 1.0f);
    normal_distribution<float> gauss_learn_spread(0.0f, 3.0f);

    Vec current_segment_center(D);
    Vec first_segment_center(D); // 用于存储第一个段的中心

    // 初始化第一个段的中心，并保存
    for (int d = 0; d < D; ++d) {
        current_segment_center[d] = gauss_center_components(eng);
        first_segment_center[d] = current_segment_center[d]; // 保存第一个中心
    }

    float shift_magnitude = 10.0f;
    int segment_size = 30000;

    learn.reserve(N_learn);
    for (int i = 0; i < N_learn; ++i) {
        // 当达到新段的起点时 (除了第一个点)，进行一次随机方向的突变
        // 注意：这里的 learn 生成逻辑与 gen_B_rand_last 完全一致，
        // first_segment_center 已经保存，不受 current_segment_center 后续变化的影响。
        if (i > 0 && i % segment_size == 0) {
            Vec random_shift_direction(D);
            float norm_sq = 0.0f;
            for (int d = 0; d < D; ++d) {
                random_shift_direction[d] = gauss_center_components(eng);
                norm_sq += random_shift_direction[d] * random_shift_direction[d];
            }
            float norm = std::sqrt(norm_sq);
            if (norm == 0) norm = 1.0f;

            for (int d = 0; d < D; ++d) {
                current_segment_center[d] += (random_shift_direction[d] / norm) * shift_magnitude;
            }
        }
        Vec v(D);
        for (int d = 0; d < D; ++d) {
            v[d] = current_segment_center[d] + gauss_learn_spread(eng);
        }
        learn.push_back(v);
    }

    // Query 集：全部在学习集 *第一个* 段的中心附近生成
    query.reserve(N_query);
    for (int i = 0; i < N_query; ++i) {
        Vec v(D);
        for (int d = 0; d < D; ++d) {
            v[d] = first_segment_center[d] + gauss_center_components(eng); // 使用第一个段的中心
        }
        query.push_back(v);
    }
}

// 生成 Pattern C：再现性转移
// 理由：模拟分布在两种状态 A、B 之间交替出现，方便研究历史模式的复用。
// learn：前半集中在状态 A，后半集中在状态 B；
// query：每 100 条在 A/B 之间切换。
void gen_C1(int D, int N_learn, int N_query, vector<Vec> &learn, vector<Vec> &query) {
    /*
     * 数据集分布：再现性转移（小偏移）
     * - Learn集：前半部分集中在状态A（N(cA, 1.0)），后半部分集中在状态B（N(cB, 1.0)）
     *           状态B相对于状态A的偏移为5.0
     * - Query集：每100个样本在状态A和状态B之间交替出现
     * 
     * 特点：模拟数据分布在两种状态之间交替出现，体现历史模式的再现性，
     *       偏移量较小（5.0），适合测试算法对历史模式的复用能力
     */
    default_random_engine eng(45);
    normal_distribution<float> gauss(0.0f, 1.0f);

    // 定义两种中心
    Vec cA(D), cB(D);
    for (int d = 0; d < D; ++d) {
        cA[d] = gauss(eng);
        cB[d] = cA[d] + 5.0f;  // B 相对于 A 的偏移
    }

    // Learn 集：前半条样本集中在 cA，后半条集中在 cB
    learn.reserve(N_learn);
    for (int i = 0; i < N_learn; ++i) {
        const Vec &c = (i < N_learn/2 ? cA : cB);
        Vec v(D);
        for (int d = 0; d < D; ++d)
            v[d] = c[d] + gauss(eng);
        learn.push_back(v);
    }

    // Query 集：每 100 条在 A/B 之间交替，体现再现性
    int block = 100;
    query.reserve(N_query);
    for (int i = 0; i < N_query; ++i) {
        const Vec &c = ((i / block) % 2 == 0 ? cA : cB);
        Vec v(D);
        for (int d = 0; d < D; ++d)
            v[d] = c[d] + gauss(eng);
        query.push_back(v);
    }
}

void gen_C2(int D, int N_learn, int N_query, vector<Vec> &learn, vector<Vec> &query) {
    /*
     * 数据集分布：再现性转移（大偏移）
     * - Learn集：前半部分集中在状态A（N(cA, 1.0)），后半部分集中在状态B（N(cB, 1.0)）
     *           状态B相对于状态A的偏移为50.0
     * - Query集：每100个样本在状态A和状态B之间交替出现
     * 
     * 特点：与C1类似，但偏移量很大（50.0），模拟两个完全不同的分布状态，
     *       测试算法在极端分布变化下的历史模式复用能力
     */
    default_random_engine eng(45);
    normal_distribution<float> gauss(0.0f, 1.0f);

    // 定义两种中心
    Vec cA(D), cB(D);
    for (int d = 0; d < D; ++d) {
        cA[d] = gauss(eng);
        cB[d] = cA[d] + 50.0f;  // B 相对于 A 的偏移
    }

    // Learn 集：前半条样本集中在 cA，后半条集中在 cB
    learn.reserve(N_learn);
    for (int i = 0; i < N_learn; ++i) {
        const Vec &c = (i < N_learn/2 ? cA : cB);
        Vec v(D);
        for (int d = 0; d < D; ++d)
            v[d] = c[d] + gauss(eng);
        learn.push_back(v);
    }

    // Query 集：每 100 条在 A/B 之间交替，体现再现性
    int block = 100;
    query.reserve(N_query);
    for (int i = 0; i < N_query; ++i) {
        const Vec &c = ((i / block) % 2 == 0 ? cA : cB);
        Vec v(D);
        for (int d = 0; d < D; ++d)
            v[d] = c[d] + gauss(eng);
        query.push_back(v);
    }
}

int main() {
    const int D = 128;
    const int N_learn = 100000;
    const int N_query = 1000;

    vector<Vec> learn, query;

    gen_G(D, N_learn, N_query, learn, query);
    write_fvecs("learn_G.fvecs", learn);
    write_fvecs("query_G.fvecs", query);
    learn.clear(); query.clear();

    // 四种模式依次生成对应的学习集和查询集，并写入 .fvecs 文件
    gen_A1(D, N_learn, N_query, learn, query);
    write_fvecs("learn_A1.fvecs", learn);
    write_fvecs("query_A1.fvecs", query);
    learn.clear(); query.clear();

    gen_A2(D, N_learn, N_query, learn, query);
    write_fvecs("learn_A2.fvecs", learn);
    write_fvecs("query_A2.fvecs", query);
    learn.clear(); query.clear();

    gen_B1(D, N_learn, N_query, learn, query);
    write_fvecs("learn_B1.fvecs", learn);
    write_fvecs("query_B1.fvecs", query);
    learn.clear(); query.clear();

    gen_B2(D, N_learn, N_query, learn, query);
    write_fvecs("learn_B2.fvecs", learn);
    write_fvecs("query_B2.fvecs", query);
    learn.clear(); query.clear();

    gen_B3(D, N_learn, N_query, learn, query);
    write_fvecs("learn_B3.fvecs", learn);
    write_fvecs("query_B3.fvecs", query);
    learn.clear(); query.clear();

    gen_B4(D, N_learn, N_query, learn, query);
    write_fvecs("learn_B4.fvecs", learn);
    write_fvecs("query_B4.fvecs", query);
    learn.clear(); query.clear();

    gen_C1(D, N_learn, N_query, learn, query);
    write_fvecs("learn_C1.fvecs", learn);
    write_fvecs("query_C1.fvecs", query);
    learn.clear(); query.clear();

    gen_C2(D, N_learn, N_query, learn, query);
    write_fvecs("learn_C2.fvecs", learn);
    write_fvecs("query_C2.fvecs", query);
    learn.clear(); query.clear();

    cout << "Generated all datasets.\n";
    return 0;
}