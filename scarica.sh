#!/bin/bash
# Scarica SOLO le immagini esistenti dal bucket e le salva con lo stesso nome.
# Destinazione: /home/luca/segments/supporti

set -u -o pipefail

BASE_URL="https://s3.tarchna.di.unimi.it/iesp-development"
DEST_DIR="/home/luca/segments/supporti"
MAX=4000

mkdir -p "$DEST_DIR"

for i in $(seq 1 "$MAX"); do
  dest="${DEST_DIR}/${i}.jpg"
  url="${BASE_URL}/${i}.jpg"

  # Se esiste già locale, salta (togli l'if se vuoi riscaricarle comunque)
  if [[ -f "$dest" ]]; then
    echo "EXIST ${i}.jpg (skip)"
    continue
  fi

  tmp="$(mktemp)"
  if curl -fsSL --retry 3 --retry-delay 1 -o "$tmp" "$url"; then
    mv "$tmp" "$dest"
    echo "OK    ${i}.jpg"
  else
    rm -f "$tmp"
    echo "SKIP  ${i}.jpg (404 o errore rete)"
  fi
done

echo "Fatto. File in: $DEST_DIR"
