#!/usr/bin/env bash
# Me ubico en el directorio de la API
cd /var/www/html/tif_api
# Se inicia el servidor local para utilizar la API
exec venv/bin/uvicorn main:app --reload