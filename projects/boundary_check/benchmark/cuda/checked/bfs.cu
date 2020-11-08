extern "C" __global__ void bfs_checked(int *ptr, int *idx, int *res_gold, int iteration, int N, int E, bool *graph_mask, bool *graph_visited, bool *updating_graph_mask) {

    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < N && graph_mask[v]) {
        graph_mask[v] = false;
        for (int i = ptr[v]; i < ptr[v + 1]; i++) {
            int id = idx[i];
            if (!graph_visited[id]) {
                res_gold[id] = iteration;
                updating_graph_mask[id] = true;
            }
        }
    }
}
