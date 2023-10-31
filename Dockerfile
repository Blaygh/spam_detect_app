# Stage 1: Build Stage
FROM python:3.10 AS build

# Set the working directory in the build stage
WORKDIR /app

# Copy the requirements.txt file from folder A to the build stage
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Stage 2: Production Stage
FROM python:3.10

# Set the working directory in the production stage
WORKDIR /app

# Copy the installed Python packages from the build stage
COPY --from=build /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/

# Copy the app.py and data files from folder B to the production stage
COPY spam_detect_app/app.py .
COPY spam_detect_app/spam_preprocessed.csv .

# Set the command to run your application
CMD ["python", "app.py"]

