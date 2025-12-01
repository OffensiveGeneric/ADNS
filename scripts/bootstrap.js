#!/usr/bin/env node
// Cross-platform bootstrap wrapper to set up local dependencies.
const { spawnSync } = require("node:child_process");
const { join, resolve } = require("node:path");
const { existsSync } = require("node:fs");

const repoRoot = resolve(__dirname, "..");
const isWindows = process.platform === "win32";
const scriptPath = isWindows
  ? join(repoRoot, "scripts", "setup_local.ps1")
  : join(repoRoot, "scripts", "setup_local.sh");

if (!existsSync(scriptPath)) {
  console.error(`Setup script not found at ${scriptPath}`);
  process.exit(1);
}

const command = isWindows ? "pwsh" : "bash";
const args = isWindows ? ["-File", scriptPath] : [scriptPath];

const result = spawnSync(command, args, {
  stdio: "inherit",
  cwd: repoRoot,
});

if (result.error) {
  console.error(`Failed to run bootstrap: ${result.error.message}`);
  process.exit(1);
}

process.exit(result.status ?? 0);
