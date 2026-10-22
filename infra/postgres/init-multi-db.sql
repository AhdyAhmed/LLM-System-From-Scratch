-- Day004: Provision isolated databases and credentials for Application, MLflow, and Airflow
-- Executed automatically on initial PostgreSQL volume creation.

CREATE ROLE llm_app_user WITH LOGIN PASSWORD 'llm_app_local_pass';
CREATE ROLE mlflow_user WITH LOGIN PASSWORD 'mlflow_local_pass';
CREATE ROLE airflow_user WITH LOGIN PASSWORD 'airflow_local_pass';

CREATE DATABASE llm_app OWNER llm_app_user;
CREATE DATABASE mlflow_db OWNER mlflow_user;
CREATE DATABASE airflow_db OWNER airflow_user;

REVOKE ALL ON DATABASE llm_app FROM PUBLIC;
REVOKE ALL ON DATABASE mlflow_db FROM PUBLIC;
REVOKE ALL ON DATABASE airflow_db FROM PUBLIC;

GRANT CONNECT, TEMPORARY ON DATABASE llm_app TO llm_app_user;
GRANT CONNECT, TEMPORARY ON DATABASE mlflow_db TO mlflow_user;
GRANT CONNECT, TEMPORARY ON DATABASE airflow_db TO airflow_user;

\connect llm_app
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;
GRANT ALL ON SCHEMA public TO llm_app_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO llm_app_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO llm_app_user;

\connect mlflow_db
GRANT ALL ON SCHEMA public TO mlflow_user;

\connect airflow_db
GRANT ALL ON SCHEMA public TO airflow_user;
