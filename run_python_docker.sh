cd ./src

docker build . --tag "pythonapp" --file ./scripts_python/DockerFile

docker run pythonapp