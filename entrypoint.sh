#!/bin/sh
export PORT=${PORT:-8000}
exec supervisord -c /app/supervisord.conf
