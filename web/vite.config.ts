import { defineConfig } from "vite";
import path from "path";
import fs from "fs";
import type { Plugin } from "vite";

function serveDataDir(): Plugin {
  const baseDir = path.resolve(__dirname, process.env.DATA_DIR || "../output");
  return {
    name: "serve-data",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        // List available datasets (subdirs containing meta.json)
        if (req.url === "/data/__list__") {
          const datasets: string[] = [];
          // Check if baseDir itself has meta.json (root dataset)
          if (fs.existsSync(path.join(baseDir, "meta.json"))) {
            datasets.push(".");
          }
          // Scan subdirectories
          for (const entry of fs.readdirSync(baseDir, { withFileTypes: true })) {
            if (entry.isDirectory() && fs.existsSync(path.join(baseDir, entry.name, "meta.json"))) {
              datasets.push(entry.name);
            }
          }
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify(datasets));
          return;
        }

        if (req.url?.startsWith("/data/")) {
          const filePath = path.join(baseDir, req.url.slice(6));
          if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
            const ext = path.extname(filePath);
            const mimeTypes: Record<string, string> = {
              ".json": "application/json",
              ".bin": "application/octet-stream",
              ".jpg": "image/jpeg",
              ".jpeg": "image/jpeg",
              ".png": "image/png",
            };
            res.setHeader("Content-Type", mimeTypes[ext] || "application/octet-stream");
            fs.createReadStream(filePath).pipe(res);
            return;
          }
        }
        next();
      });
    },
  };
}

export default defineConfig({
  plugins: [serveDataDir()],
  server: {
    fs: {
      allow: [path.resolve(__dirname, "..")],
    },
  },
  publicDir: false,
});
