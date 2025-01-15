#!/bin/bash

# Check if the config directory exists and has any yaml files
if [ ! -d "/code/config" ] || [ -z "$(ls -A /code/config/*.yaml 2>/dev/null)" ]; then
    echo "Error: No configuration files found in /code/config/"
    exit 1
else
    echo "Configuration files found in /code/config/"
fi

# Execute the provided command
exec "$@"