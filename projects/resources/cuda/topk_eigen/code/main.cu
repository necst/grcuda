#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

struct CooMatrix {
    vector<int> x;
    vector<int> y;
    vector<float> val;
};

struct CooMatrix *readDataset(string path, bool normalize = false);

// JACOBI alg
class Jacobi {
public:
    Jacobi(int eigenCount) {
        N = eigenCount;
    }

    Jacobi *build(const float tm[]) {
        float matrix[N][N];
        float originalMatrix[N][N];
        float eigenvectors[N][N];
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                matrix[r][c] = 0;
                originalMatrix[r][c] = 0;
                eigenvectors[r][c] = 0;
            }
        }
        // START - private method: _asDenseMatrix()
        for (int i = 0; i < N - 1; i++) {
            float alpha = tm[i << 1];
            float beta = tm[(i << 1) + 1];
            matrix[i][i] = alpha;
            matrix[i + 1][i] = beta;
            matrix[i][i + 1] = beta;
        }
        // There is a warning here -> sizeof(tm), it isn't taking the len(tm)
        matrix[N - 1][N - 1] = tm[(sizeof(tm) / sizeof(tm[0])) - 1];
        for (int i = 0; i < N; i++) {
            eigenvectors[i][i] = 1.0;
        }
        // END - private method: _asDenseMatrix()
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                originalMatrix[i][j] = matrix[i][j];
            }
        }
        return this;
    }

private:
    int N;

};

int main(int argc, char *argv[]) {
    struct CooMatrix cooMatrix = *readDataset(argv[2]);
    int numEigen = stoi(argv[3]);
    int numRuns = stoi(argv[4]);
    int numPartitions = stoi(argv[5]);
    bool reorthogonalize = (strcmp(argv[6], "true") == 0);
    bool debug = (strcmp(argv[7], "true") == 0);

    // Check for it
    string input = (argc > 7) ? argv[7] : "float";
    string middle = (argc > 8) ? argv[8] : "float";
    string output = (argc > 9) ? argv[9] : "float";

    if (debug) cout << input + " - " + middle + " - " + output;

    // COMMAND -> lanczosKernel.build().transferToGPU()

    for (int i = 0; i < numRuns; i++) {
        // RUN LANCZOS ( lanczosKernel.compute(i).printResults() )

        // RUN JACOBI ( jacobiKernel.build(lanczosKernel.tridiagonalMatrix).compute().printResults() )

        // RUN LANCZOS ( lanczosKernel.computeFullReconstructionError(cooMatrix, lanczosKernel.lanczosVectors, jacobiKernel.matrix, jacobiKernel.eigenvectors) )

        // RESET (
        //          lanczosKernel.reset()
        //          jacobiKernel.reset()
        //        )
    }
}

void readDataset(CooMatrix *result, string path, bool normalize) {
    ifstream file;
    file.open(path);
    while (file) {
        string line;
        getline(file, line);
        if (line.find('%') == string::npos) {
            stringstream tmpLine(line);
            string val;
            int index = 0;
            while (getline(tmpLine, val, ' ')) {
                if (index == 0) result->x.push_back(stoi(val));
                else if (index == 1) result->y.push_back((stoi(val) - 1));
                else if (index == 2) result->val.push_back(stof(val));
                index++;
            }
            if (index == 1) {
                cerr << "Error" << "\n";
                break;
            } else if (index == 2) {
                result->val.push_back(1);
            }
        }
        if (normalize) {
            /*
             * const norm = Math.sqrt(val.reduce((acc, cur) => acc + cur * cur))
             * if(normalize) val = val.map(value => value / norm)
             */
        }
    }
    file.close();
}