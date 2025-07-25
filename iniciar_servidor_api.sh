#!/usr/bin/env bash
# Me ubico en el directorio de la API
cd /var/www/html/tif_api
# Se inicia el servidor local para utilizar la API
exec venv/bin/uvicorn main:app --reload

#sudo -u www-data bash -lc \
#  "cd /var/www/html/tif_api && venv/bin/uvicorn main:app --reload > /var/log/uvicorn.log 2>&1 & echo \$!" \
#  > /run/uvicorn.pid