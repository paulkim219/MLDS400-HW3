cd ./src

docker build . --tag "rapp" --file ./scripts_r/DockerFile

docker run -v "$(pwd)/data:/data" rapp