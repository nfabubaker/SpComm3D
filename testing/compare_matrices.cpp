#include "compare_mtx.hpp"

const double tol = 1e-6;

// To add a test, use the following syntax (replace mtx filenames, tolerance and test name as needed):
// TEST(MatrixComparison, Simple) {
//     std::string path_A = "matrices/simple.mtx";
//     std::string path_B = "matrices/simple_ref.mtx";

//     test_eq_mtx(path_A, path_B, tol);
// }

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}