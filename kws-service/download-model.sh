#!/bin/sh
set -eu

MODEL_NAME="sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
MODEL_ROOT="${KWS_MODEL_ROOT:-/models}"
MODEL_DIR="$MODEL_ROOT/$MODEL_NAME"
ARCHIVE="$MODEL_ROOT/$MODEL_NAME.tar.bz2"
URL="${KWS_MODEL_URL:-https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/$MODEL_NAME.tar.bz2}"
EXPECTED_SHA256="${KWS_MODEL_SHA256:-b2f7c89690dc8ce4c6ed6afeab7cd800c36ad1421fb6b6302b4a4b194cf7f35f}"

if [ -f "$MODEL_DIR/tokens.txt" ] \
  && [ -f "$MODEL_DIR/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx" ] \
  && [ -f "$MODEL_DIR/decoder-epoch-12-avg-2-chunk-16-left-64.onnx" ] \
  && [ -f "$MODEL_DIR/joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx" ]; then
  exit 0
fi

mkdir -p "$MODEL_ROOT"
rm -f "$ARCHIVE"
curl --fail --location --retry 3 --output "$ARCHIVE" "$URL"
printf '%s  %s\n' "$EXPECTED_SHA256" "$ARCHIVE" | sha256sum --check -
tar -xjf "$ARCHIVE" -C "$MODEL_ROOT"
rm -f "$ARCHIVE"
