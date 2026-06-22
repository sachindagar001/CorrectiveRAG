#!/bin/bash
# Persistent startup script for CRAG backend services
# Keeps both Next.js and FastAPI running in the background

cd /Users/raman/Downloads/Projects/crag-agent-source

# Kill any existing instances
pkill -9 -f "server.py" 2>/dev/null
pkill -9 -f "next dev" 2>/dev/null
sleep 2

# Activate venv for Python
source venv/bin/activate

# Start FastAPI backend in a new session
venv/bin/python mini-services/crag-api/server.py > /tmp/crag_api.log 2>&1 < /dev/null &
disown
echo "FastAPI started (port 8000)"

# Wait for FastAPI to be ready
for i in 1 2 3 4 5 6 7 8 9 10; do
    if curl -s --max-time 2 http://localhost:8000/api/health > /dev/null 2>&1; then
        echo "FastAPI ready"
        break
    fi
    sleep 1
done

# Start Next.js dev server (if not already running)
if ! curl -s --max-time 2 http://localhost:3000 > /dev/null 2>&1; then
    ./node_modules/.bin/next dev -p 3000 -H 0.0.0.0 > /tmp/nextjs.log 2>&1 < /dev/null &
    disown
    echo "Next.js started (port 3000)"
    # Wait for Next.js
    for i in 1 2 3 4 5 6 7 8 9 10; do
        if curl -s --max-time 2 http://localhost:3000 > /dev/null 2>&1; then
            echo "Next.js ready"
            break
        fi
        sleep 2
    done
else
    echo "Next.js already running"
fi

# Final status
echo ""
echo "=== Status ==="
curl -s -o /dev/null -w "Next.js (3000): HTTP %{http_code}\n" --max-time 5 http://localhost:3000
curl -s -o /dev/null -w "FastAPI (8000): HTTP %{http_code}\n" --max-time 5 http://localhost:8000/api/health
