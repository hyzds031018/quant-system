#!/bin/bash
echo "Starting installation script..."

# Define the port
PORT=${PORT:-8080}

# 1. Start a simple Python web server IMMEDIATELY to satisfy Cloud Run's port binding requirement
echo "Starting dummy server on port $PORT to satisfy Cloud Run health check..."
python3 -c "
import http.server, socketserver, os
PORT = int(os.environ.get('PORT', 8080))
class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'Application is starting up. Installing dependencies... Please refresh in a minute.')
httpd = socketserver.TCPServer(('', PORT), Handler)
httpd.serve_forever()
" &
DUMMY_PID=$!

echo "Dummy server started with PID $DUMMY_PID"

# 2. Install PyTorch in the foreground (this might take 1-2 minutes)
echo "Installing CPU-only PyTorch..."
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 3. Kill the dummy server
echo "Installation complete. Killing dummy server..."
kill $DUMMY_PID
wait $DUMMY_PID 2>/dev/null

# 4. Start the actual Gunicorn application
echo "Starting Gunicorn application..."
exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
