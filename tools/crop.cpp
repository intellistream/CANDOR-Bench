#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

std::pair<std::vector<float>, int> read_all_fvecs(const char *filename) {
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        perror("open input file");
        exit(-1);
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("fstat");
        exit(-1);
    }

    char *fileptr =
        static_cast<char *>(mmap(0, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (fileptr == MAP_FAILED) {
        perror("mmap");
        exit(-1);
    }
    close(fd);

    int dim = *reinterpret_cast<int *>(fileptr);
    size_t vector_size_bytes = sizeof(int) + dim * sizeof(float);
    size_t total_vectors = sb.st_size / vector_size_bytes;

    std::vector<float> data;
    data.reserve(total_vectors * dim);

    char *ptr = fileptr;
    for (size_t i = 0; i < total_vectors; ++i) {
        int current_dim = *reinterpret_cast<int *>(ptr);
        if (current_dim != dim) {
            std::cerr << "Dimension mismatch at vector " << i << std::endl;
            break;
        }
        ptr += sizeof(int);
        float *vec = reinterpret_cast<float *>(ptr);
        data.insert(data.end(), vec, vec + dim);
        ptr += dim * sizeof(float);
    }

    munmap(fileptr, sb.st_size);
    return {data, dim};
}

std::pair<std::vector<float>, int> read_all_bin(const char *filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open bin file: " << filename << std::endl;
        exit(-1);
    }

    int npts, dim;
    file.read(reinterpret_cast<char *>(&npts), sizeof(int));
    file.read(reinterpret_cast<char *>(&dim), sizeof(int));

    std::vector<float> data;
    data.reserve(npts * dim);

    for (int i = 0; i < npts; ++i) {
        std::vector<float> vec(dim);
        file.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(float));
        data.insert(data.end(), vec.begin(), vec.end());
    }

    file.close();
    return {data, dim};
}

std::vector<size_t> read_indices_from_file(const char *filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open index file: " << filename << std::endl;
        exit(-1);
    }

    std::vector<size_t> indices;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string token;

        while (std::getline(ss, token, ',')) {
            if (!token.empty()) {
                try {
                    size_t idx = std::stoul(token);
                    indices.push_back(idx);
                } catch (const std::exception &e) {
                    std::cerr << "Invalid index: " << token << std::endl;
                }
            }
        }
    }

    file.close();
    return indices;
}

void write_fvecs(const char *filename, const std::vector<float> &data, int n,
                 int dim) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        perror("open output file");
        exit(-1);
    }

    for (int i = 0; i < n; ++i) {
        out.write(reinterpret_cast<const char *>(&dim), sizeof(int));
        out.write(reinterpret_cast<const char *>(data.data() + i * dim),
                  dim * sizeof(float));
    }
    out.close();
}

void write_bin(const char *filename, const std::vector<float> &data, int n,
               int dim) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        perror("open output file");
        exit(-1);
    }

    out.write(reinterpret_cast<const char *>(&n), sizeof(int));
    out.write(reinterpret_cast<const char *>(&dim), sizeof(int));
    out.write(reinterpret_cast<const char *>(data.data()),
              n * dim * sizeof(float));
    out.close();
}

int main(int argc, char *argv[]) {
    if (argc < 5 || argc > 6) {
        std::cout << "Usage: " << argv[0]
                  << " <input_file> <num_vectors> <output_file> <shuffle> "
                     "[index_file.txt]"
                  << std::endl;
        std::cout << "  input_file: .fvecs or .bin file" << std::endl;
        std::cout << "  output_file: .fvecs or .bin file (format matches input)"
                  << std::endl;
        std::cout << "  shuffle: 1 to shuffle vectors, 0 to keep original order"
                  << std::endl;
        std::cout << "  index_file.txt: optional file containing "
                     "comma-separated indices"
                  << std::endl;
        return 1;
    }

    const char *input_file = argv[1];
    int n = std::atoi(argv[2]);
    const char *output_file = argv[3];
    bool shuffle = std::atoi(argv[4]) != 0;
    const char *index_file = argc == 6 ? argv[5] : nullptr;

    // Detect file format based on extension
    std::string input_str(input_file);
    std::string output_str(output_file);
    bool is_fvecs =
        input_str.substr(input_str.find_last_of(".") + 1) == "fvecs";
    bool output_is_fvecs =
        output_str.substr(output_str.find_last_of(".") + 1) == "fvecs";

    if (n <= 0) {
        std::cerr << "Number of vectors must be positive" << std::endl;
        return 1;
    }

    std::pair<std::vector<float>, int> data_result;
    if (is_fvecs) {
        data_result = read_all_fvecs(input_file);
    } else {
        data_result = read_all_bin(input_file);
    }
    auto [all_data, dim] = data_result;
    size_t total_vectors = all_data.size() / dim;

    std::cout << "Read " << total_vectors << " vectors with dimension " << dim
              << " from " << input_file << std::endl;

    if (n > total_vectors) {
        std::cerr << "Requested " << n << " vectors, but file only has "
                  << total_vectors << std::endl;
        n = total_vectors;
    }

    std::vector<float> selected_data;
    selected_data.reserve(n * dim);

    if (index_file) {
        std::vector<size_t> indices = read_indices_from_file(index_file);
        std::cout << "Read " << indices.size() << " indices from " << index_file
                  << std::endl;

        if (indices.size() > static_cast<size_t>(n)) {
            indices.resize(n);
        }

        for (size_t idx : indices) {
            if (idx >= total_vectors) {
                std::cerr << "Warning: Index " << idx
                          << " is out of range (max: " << total_vectors - 1
                          << "), skipping" << std::endl;
                continue;
            }
            selected_data.insert(selected_data.end(),
                                 all_data.begin() + idx * dim,
                                 all_data.begin() + (idx + 1) * dim);
        }
        std::cout << "Selected " << selected_data.size() / dim
                  << " vectors using indices from file" << std::endl;
    } else if (shuffle) {
        std::vector<size_t> indices(total_vectors);
        std::iota(indices.begin(), indices.end(), 0);

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        for (int i = 0; i < n; ++i) {
            size_t idx = indices[i];
            selected_data.insert(selected_data.end(),
                                 all_data.begin() + idx * dim,
                                 all_data.begin() + (idx + 1) * dim);
        }
        std::cout << "Shuffled and selected " << n << " vectors" << std::endl;
    } else {
        for (int i = 0; i < n; ++i) {
            selected_data.insert(selected_data.end(),
                                 all_data.begin() + i * dim,
                                 all_data.begin() + (i + 1) * dim);
        }
        std::cout << "Selected first " << n << " vectors in original order"
                  << std::endl;
    }

    int actual_n = selected_data.size() / dim;
    if (output_is_fvecs) {
        write_fvecs(output_file, selected_data, actual_n, dim);
    } else {
        write_bin(output_file, selected_data, actual_n, dim);
    }
    std::cout << "Wrote " << actual_n << " vectors to " << output_file
              << std::endl;

    return 0;
}