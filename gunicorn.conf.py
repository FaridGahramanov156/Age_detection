# gunicorn.conf.py
# Gunicorn configuration file for Age Prediction App

import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
backlog = 2048

# Worker processes
workers = 1  # Single worker for face learning model consistency
worker_class = "sync"
worker_connections = 1000
timeout = 120  # Increased timeout for model initialization
keepalive = 2
# Restart workers
max_requests = 1000
max_requests_jitter = 50
preload_app = True  # Preload app for better performance

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = "age_prediction_app"

# Server mechanics
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

# SSL (not needed for Render, but can be configured)
keyfile = None
certfile = None