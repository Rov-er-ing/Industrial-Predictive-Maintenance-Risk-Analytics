# Setup Project Script

# Create graphs directory
if (!(Test-Path ".planning\graphs")) {
    New-Item -ItemType Directory -Path ".planning\graphs" -Force
}

# Run graphify
# Note: graphify might need to be in the PATH
graphify . --update

# Copy artifacts if graphify-out exists
if (Test-Path "graphify-out") {
    Copy-Item -Path "graphify-out\graph.json" -Destination ".planning\graphs\graph.json" -ErrorAction SilentlyContinue
    Copy-Item -Path "graphify-out\graph.html" -Destination ".planning\graphs\graph.html" -ErrorAction SilentlyContinue
    Copy-Item -Path "graphify-out\GRAPH_REPORT.md" -Destination ".planning\graphs\GRAPH_REPORT.md" -ErrorAction SilentlyContinue
}

# Run snapshot
node "C:\Users\Shiti\.gemini\antigravity\get-shit-done\bin\gsd-tools.cjs" graphify build snapshot
