# Web Demo for SeptembeRSE
The goal of this demo is to showcase an image processing pipeline in _GrCUDA_.

## Backend
The backend is in charge of receiving signal (via `websockets`) associated to the beginning of the computation and the computation mode (either `sync`, `async` or `cuda-native`) from the frontend and initiate the actual computation using the specified mode.
At each image processed, the backend signals the frontend of the current progresses and of which images (in batch) are ready to be displayed to the final user.

**_NOTE: currently, the backend mocks the actual GrCUDA computation_**

### Install dependencies and run
To install the dependencies run either `yarn install` or `npm install` in the `backend` directory.
When in development, it is advisable to run `yarn run dev` in order to compile and run the server at each code save.
In production, first compile the `typescript` files using the `typescript` compiler (`tsc`). The compiled files can be found in the `dist` directory and can be executed by running `node .`.


## Frontend
The frontend is in charge of signaling the beginning of the computation to the backend, showing the progress and, when the computation is finished, display a grid of the computed images. By clicking on any thumbnail in the grid, the user is displayed the full resolution image.

### Install dependencies and run
Open the `index.html` file, requires the backend to be already running in the local server (`localhost`) on port 8080.