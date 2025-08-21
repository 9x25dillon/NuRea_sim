FROM ruby:3.3

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy Ruby files
COPY . .

# Install Ruby dependencies
RUN bundle install

# Make kiln executable
RUN chmod +x bin/kiln

EXPOSE 3000

CMD ["/bin/sh", "-lc", "sleep infinity"]
