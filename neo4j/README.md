### Access to docker contrainer
    docker exec -it <container-id> bash

### Show all dtabases
    cypher-shell -u neo4j -p p@ssw0rd -d system "SHOW DATABASES;"

    docker exec -it <container-id> sh -c 'cypher-shell -u neo4j -p p@ssw0rd "SHOW DATABASES"'

### Access to database
    cypher-shell -u neo4j -p p@ssw0rd

### Create database
    CREATE DATABASE <database-name>

- If you run them in the Community Edition, you will get the message “Unsupported administrative command”.
    Neo.ClientError.Statement.UnsupportedAdministrationCommand:
    Unsupported administration command: create database test