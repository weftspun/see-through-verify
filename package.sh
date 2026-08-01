#!/bin/bash
# Package see-through CLI as macOS .app bundle
set -euo pipefail

APP="/tmp/see-through.app"
BUNDLE="$APP/Contents"

rm -rf "$APP"
mkdir -p "$BUNDLE/MacOS" "$BUNDLE/Resources" "$BUNDLE/SharedSupport"

echo "=== Building CLI ==="
clang++ -std=c++17 Runtime/see-through.cpp -o "$BUNDLE/MacOS/see-through" -O2

echo "=== Info.plist ==="
cat > "$BUNDLE/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key>
  <string>see-through</string>
  <key>CFBundleIdentifier</key>
  <string>com.weftspun.see-through</string>
  <key>CFBundleName</key>
  <string>See-Through</string>
  <key>CFBundleVersion</key>
  <string>0.1.0</string>
  <key>CFBundleShortVersionString</key>
  <string>0.1.0</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>LSMinimumSystemVersion</key>
  <string>14.0</string>
</dict>
</plist>
EOF

echo "=== Bundling weights ==="
if [ -d hf_cache ] && [ "$(find hf_cache -name '*.safetensors' 2>/dev/null | wc -l)" -gt 0 ]; then
  mkdir -p "$BUNDLE/SharedSupport/hf_cache"
  cp -r hf_cache/* "$BUNDLE/SharedSupport/hf_cache/"
  echo "  weights: $(du -sh "$BUNDLE/SharedSupport/hf_cache" | cut -f1)"
else
  echo "  no weights found"
fi

echo "=== Setting environment ==="
# Create a launcher script that sets SEE_THROUGH_DIR to the bundled weights
cat > "$BUNDLE/MacOS/see-through-launcher" <<'LAUNCHER'
#!/bin/bash
DIR="$(cd "$(dirname "$0")/../SharedSupport" && pwd)"
export SEE_THROUGH_DIR="$DIR/hf_cache"
exec "$(dirname "$0")/see-through" "$@"
LAUNCHER
chmod +x "$BUNDLE/MacOS/see-through-launcher"

echo "=== Creating .app ==="
cp -r "$APP" .
echo "Done: $(du -sh see-through.app | cut -f1)"