uv init name
uv add xx
#删除后执行
uv sync 
ollama serve
ollama ls
ollama run
# docker 安装postgresql
docker run -d --name postgres-db -p 5432:5432 -e POSTGRES_PASSWORD=123456 -e POSTGRES_USER=postgres -e POSTGRES_DB=my_db -v postgres-data:/var/lib/postgresql/data postgres:16