# To pull pytorch image
docker pull pytorch/pytorch

# To run shell on docker image with PWD mounted in /workspace
docker run -it --entrypoint /bin/bash -v ${PWD}:/workspace pytorch/pytorch