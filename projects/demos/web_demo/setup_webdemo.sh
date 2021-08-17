#!/bin/bash

# install cmake (required for opencv4nodejs)
sudo apt install cmake

# Create cloned directory and clone grcuda repo
mkdir Cloned
cd ~/Cloned/
git clone https://github.com/AlbertoParravicini/grcuda.git
cd ~/Cloned/grcuda
git checkout GRCUDA-36-web-demo


# Move to project directory (backend)
cd ~/Cloned/grcuda/projects/demos/web_demo/backend

# install project dependencies
npm i

# Now we should be ready. build and start the server ...
# npm run build

# node --polyglot --jvm  --vm.Dtruffle.class.path.append=../../mxbuild/dists/jdk1.8/grcuda.jar dist/index.js &

# Start the web interface
# cd ~/Cloned/grcuda/projects/demos/web_demo/frontend
# python3 -m http.server &
