cd ./src

docker build . --tag "pythonapp" --file ./scripts_python/DockerFile

docker run -v "$(pwd)/data:/data" pythonapp