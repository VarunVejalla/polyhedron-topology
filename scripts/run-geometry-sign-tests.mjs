import fs from "node:fs";
import path from "node:path";
import { execSync } from "node:child_process";

const outDir = path.resolve(".tmp_geometry_sign_tests");

function run(command) {
  execSync(command, { stdio: "inherit" });
}

try {
  run(
    [
      "npx tsc",
      "tests/geometrySignInvariants.ts",
      "--module commonjs",
      "--moduleResolution node",
      "--target es2022",
      "--outDir .tmp_geometry_sign_tests",
      "--skipLibCheck",
      "--lib ES2022,DOM",
      "--resolveJsonModule",
      "--noEmit false",
    ].join(" ")
  );

  fs.writeFileSync(path.join(outDir, "package.json"), JSON.stringify({ type: "commonjs" }), "utf8");
  run("node .tmp_geometry_sign_tests/tests/geometrySignInvariants.js");
} finally {
  fs.rmSync(outDir, { recursive: true, force: true });
}
