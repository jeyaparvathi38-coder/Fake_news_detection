import sys
import os

# Make sure the app package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.app import app

# Vercel expects a WSGI-compatible handler named 'handler' or 'app'
handler = app
