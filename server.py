#!/usr/bin/env python3
"""
Simple HTTP Server to serve the Tranche 2x Stock Screener Web Dashboard.
Automatically falls back to alternative ports if port 8000 is occupied.
Run: python3 server.py [port]
"""

import http.server
import socketserver
import sys
import os

PREFERRED_PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def start_server():
    ports_to_try = [PREFERRED_PORT, 8001, 8080, 8085, 8888]
    for port in ports_to_try:
        try:
            with socketserver.TCPServer(("", port), Handler) as httpd:
                print(f"\n🚀 Tranche 2x Stock Screener Dashboard is live at: http://localhost:{port}")
                print("Press Ctrl+C to stop the server.\n")
                httpd.serve_forever()
                return
        except OSError as e:
            if e.errno == 48: # Address in use
                continue
            else:
                raise e
    print("Error: Could not bind to any of the ports:", ports_to_try)

if __name__ == "__main__":
    start_server()
