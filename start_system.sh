#!/bin/bash

# Build the Docker image
docker build -t orvprojekt_image .

# Run the Docker container
docker run -d -p 80:80 --name orvprojekt_container orvprojekt_image