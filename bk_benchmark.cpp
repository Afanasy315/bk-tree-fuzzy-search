#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

struct Node {
    std::string word;
    std::map<int, int> children;

    Node(const std::string &word) {
        this->word = word;
    }
};

int LevenshteinDistance(const std::string &a, const std::string &b) {
    int n = static_cast<int>(a.size());
    int m = static_cast<int>(b.size());

    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1));

    for (int i = 0; i <= n; ++i) {
        dp[i][0] = i;
    }

    for (int j = 0; j <= m; ++j) {
        dp[0][j] = j;
    }

    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            int cost = (a[i - 1] == b[j - 1]) ? 0 : 1;

            dp[i][j] = std::min({
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            });
        }
    }

    return dp[n][m];
}

class BKTree {
private:
    std::vector<Node> tree;
    int root;

    void SearchDfs(int vertex, const std::string &query, int k, int &visited, int &found) const {
        ++visited;

        int dist = LevenshteinDistance(tree[vertex].word, query);

        if (dist <= k) {
            ++found;
        }

        int left = dist - k;
        int right = dist + k;

        auto it = tree[vertex].children.lower_bound(left);

        while (it != tree[vertex].children.end() && it->first <= right) {
            SearchDfs(it->second, query, k, visited, found);
            ++it;
        }
    }

    int GetMaxDepthDfs(int vertex) const {
        int result = 1;

        for (const auto &child : tree[vertex].children) {
            result = std::max(result, 1 + GetMaxDepthDfs(child.second));
        }

        return result;
    }

public:
    BKTree() {
        root = -1;
        tree.clear();
    }

    void AddWord(const std::string &word) {
        if (root == -1) {
            tree.push_back(Node(word));
            root = 0;
            return;
        }

        int current = root;

        while (true) {
            int dist = LevenshteinDistance(word, tree[current].word);

            if (tree[current].children.count(dist) == 0) {
                tree.push_back(Node(word));
                int new_vertex = static_cast<int>(tree.size()) - 1;
                tree[current].children[dist] = new_vertex;
                return;
            }

            current = tree[current].children[dist];
        }
    }

    void Search(const std::string &query, int k, int &visited, int &found) const {
        visited = 0;
        found = 0;

        if (root == -1) {
            return;
        }

        SearchDfs(root, query, k, visited, found);
    }

    double GetAverageBranching() const {
        int vertices_with_children = 0;
        int total_children = 0;

        for (const Node &node : tree) {
            if (!node.children.empty()) {
                ++vertices_with_children;
                total_children += static_cast<int>(node.children.size());
            }
        }

        if (vertices_with_children == 0) {
            return 0.0;
        }

        return static_cast<double>(total_children) / vertices_with_children;
    }

    int GetMaxDepth() const {
        if (root == -1) {
            return 0;
        }

        return GetMaxDepthDfs(root);
    }

    int GetNodeCount() const {
        return static_cast<int>(tree.size());
    }
};

void NaiveSearch(const std::vector<std::string> &dictionary,
                 const std::string &query, int k,
                 int &visited, int &found) {
    visited = 0;
    found = 0;

    for (const std::string &word : dictionary) {
        ++visited;
        if (LevenshteinDistance(word, query) <= k) {
            ++found;
        }
    }
}

std::string GenerateWord(std::mt19937 &generator) {
    std::uniform_int_distribution<int> length_distribution(5, 10);
    std::uniform_int_distribution<int> letter_distribution(0, 25);

    int length = length_distribution(generator);
    std::string word;

    for (int i = 0; i < length; ++i) {
        char letter = static_cast<char>('a' + letter_distribution(generator));
        word.push_back(letter);
    }

    return word;
}

std::vector<std::string> GenerateDictionary(int n, std::mt19937 &generator) {
    std::unordered_set<std::string> used;
    std::vector<std::string> words;

    while (static_cast<int>(words.size()) < n) {
        std::string word = GenerateWord(generator);

        if (used.count(word) == 0) {
            used.insert(word);
            words.push_back(word);
        }
    }

    return words;
}

std::vector<std::string> GenerateQueries(int count, std::mt19937 &generator) {
    std::vector<std::string> queries;

    for (int i = 0; i < count; ++i) {
        queries.push_back(GenerateWord(generator));
    }

    return queries;
}

int main() {
    std::vector<int> dictionary_sizes = {1000, 5000, 10000, 50000, 100000};
    std::vector<int> k_values = {0, 1, 2, 3};

    int query_count = 100;

    std::mt19937 generator(42);

    std::ofstream file("bk_benchmark.csv");

    file << "N,build_time_us,node_count,avg_branching,max_depth,naive_time_us";
    for (int k : k_values) {
        if (k == 0) continue;
        file << ",bk_k" << k << "_time_us,bk_k" << k << "_visited,bk_k" << k << "_found,bk_k" << k << "_speedup";
    }
    file << ",bk_k0_time_us,bk_k0_visited,bk_k0_found\n";

    for (int n : dictionary_sizes) {
        std::cout << "\n=== N = " << n << " ===\n";

        std::vector<std::string> dictionary = GenerateDictionary(n, generator);
        std::vector<std::string> queries = GenerateQueries(query_count, generator);

        auto build_start = std::chrono::high_resolution_clock::now();

        BKTree tree;
        for (const std::string &word : dictionary) {
            tree.AddWord(word);
        }

        auto build_finish = std::chrono::high_resolution_clock::now();

        long long build_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            build_finish - build_start
        ).count();

        int node_count = tree.GetNodeCount();
        double avg_branching = tree.GetAverageBranching();
        int max_depth = tree.GetMaxDepth();

        std::cout << "build_time_us: " << build_time_us << "\n"
                  << "node_count: " << node_count
                  << ", avg_branching: " << avg_branching
                  << ", max_depth: " << max_depth << "\n";

        long long naive_total_time = 0;

        for (const std::string &query : queries) {
            int visited_dummy, found_dummy;
            auto start = std::chrono::high_resolution_clock::now();
            NaiveSearch(dictionary, query, 0, visited_dummy, found_dummy);
            auto finish = std::chrono::high_resolution_clock::now();

            naive_total_time += std::chrono::duration_cast<std::chrono::microseconds>(
                finish - start
            ).count();
        }

        double naive_avg_time_us = static_cast<double>(naive_total_time) / query_count;

        std::cout << "naive_avg_time_us: " << naive_avg_time_us << "\n";

        std::vector<double> bk_times(4, 0.0);
        std::vector<double> bk_visited(4, 0.0);
        std::vector<double> bk_found(4, 0.0);

        for (int k : k_values) {
            long long total_time = 0;
            long long total_visited = 0;
            long long total_found = 0;

            for (const std::string &query : queries) {
                int visited = 0;
                int found = 0;

                auto start = std::chrono::high_resolution_clock::now();

                tree.Search(query, k, visited, found);

                auto finish = std::chrono::high_resolution_clock::now();

                long long time_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    finish - start
                ).count();

                total_time += time_us;
                total_visited += visited;
                total_found += found;
            }

            bk_times[k] = static_cast<double>(total_time) / query_count;
            bk_visited[k] = static_cast<double>(total_visited) / query_count;
            bk_found[k] = static_cast<double>(total_found) / query_count;

            std::cout << "k=" << k
                      << " time_us: " << bk_times[k]
                      << " visited: " << bk_visited[k]
                      << " found: " << bk_found[k] << "\n";
        }

        file << n << "," << build_time_us << "," << node_count << "," << avg_branching << "," << max_depth;
        file << "," << naive_avg_time_us;

        for (int k : {1, 2, 3}) {
            double speedup = naive_avg_time_us / bk_times[k];
            file << "," << bk_times[k] << "," << bk_visited[k] << "," << bk_found[k] << "," << speedup;
        }

        file << "," << bk_times[0] << "," << bk_visited[0] << "," << bk_found[0] << "\n";
    }

    file.close();

    std::cout << "\nResults saved to bk_benchmark.csv\n";

    return 0;
}
