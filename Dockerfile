# Use Ubuntu as the base image for a stable C++ environment
FROM ubuntu:22.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build essentials and OpenMP
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the project files into the container
COPY . .

# Compile the application
# -O3: Maximum optimization
# -fopenmp: Enables OpenMP for parallelization
# -msse4.1: Ensures SSE 4.1 instructions are available
# -I engine: Adds the engine directory to the include path
RUN g++ -O3 -fopenmp -msse4.1 engine/run.cpp -I engine -o tinygpt_server

# Render uses the PORT environment variable, but your code uses 8080.
EXPOSE 8080

# Run the server
CMD ["./tinygpt_server"]
