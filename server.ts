import express from 'express';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

  app.use(express.json({ limit: '50mb' }));
  app.use(express.urlencoded({ extended: true, limit: '50mb' }));

  const UNMIX_DIR = path.resolve(__dirname, 'unmix-the-signal');
  const OUTPUTS_DIR = path.resolve(UNMIX_DIR, 'outputs');
  const AUDIO_DIR = path.resolve(OUTPUTS_DIR, 'audio');
  const PLOTS_DIR = path.resolve(OUTPUTS_DIR, 'plots');

  // API 1: Results Summary
  app.get('/api/summary', (req, res) => {
    res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
    const summaryFile = path.resolve(OUTPUTS_DIR, 'results_summary.json');
    if (fs.existsSync(summaryFile)) {
      try {
        const data = JSON.parse(fs.readFileSync(summaryFile, 'utf-8'));
        return res.json(data);
      } catch (err) {
        return res.status(500).json({ error: 'Failed to read results summary.' });
      }
    } else {
      return res.status(404).json({ error: 'Summary not generated yet.' });
    }
  });

  // API 2: Stream & Download Audio Files
  app.get('/api/audio/:filename', (req, res) => {
    const filename = req.params.filename;
    // Prevent directory traversal
    const safeFilename = path.basename(filename);
    const filePath = path.resolve(AUDIO_DIR, safeFilename);

    if (!fs.existsSync(filePath)) {
      return res.status(404).send('Audio file not found');
    }

    const stat = fs.statSync(filePath);
    const fileSize = stat.size;
    const range = req.headers.range;

    res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
    res.setHeader('Pragma', 'no-cache');
    res.setHeader('Expires', '0');

    if (range) {
      const parts = range.replace(/bytes=/, "").split("-");
      const start = parseInt(parts[0], 10);
      const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
      const chunksize = (end - start) + 1;
      const file = fs.createReadStream(filePath, { start, end });
      const head = {
        'Content-Range': `bytes ${start}-${end}/${fileSize}`,
        'Accept-Ranges': 'bytes',
        'Content-Length': chunksize,
        'Content-Type': 'audio/wav',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0',
      };
      res.writeHead(206, head);
      file.pipe(res);
    } else {
      const head: Record<string, any> = {
        'Content-Length': fileSize,
        'Content-Type': 'audio/wav',
        'Accept-Ranges': 'bytes',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0',
      };
      if (req.query.download === 'true') {
        head['Content-Disposition'] = `attachment; filename="${safeFilename}"`;
      }
      res.writeHead(200, head);
      fs.createReadStream(filePath).pipe(res);
    }
  });

  // API 3: Serve Generated Plots
  app.get('/api/plots/:filename', (req, res) => {
    const safeFilename = path.basename(req.params.filename);
    const filePath = path.resolve(PLOTS_DIR, safeFilename);
    if (fs.existsSync(filePath)) {
      res.setHeader('Content-Type', 'image/png');
      res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
      res.setHeader('Pragma', 'no-cache');
      res.setHeader('Expires', '0');
      fs.createReadStream(filePath).pipe(res);
    } else {
      res.status(404).send('Plot not found');
    }
  });

  // Health check endpoint
  app.get('/api/health', (req, res) => {
    res.setHeader('Cache-Control', 'no-cache');
    res.json({
      status: 'ok',
      timestamp: Date.now(),
      backend: 'express-python-fastica',
      audio_dir_exists: fs.existsSync(AUDIO_DIR),
      plots_dir_exists: fs.existsSync(PLOTS_DIR)
    });
  });

  // Helper to safely extract JSON from stdout containing potential warnings
  function extractJson(raw: string): any {
    const firstBrace = raw.indexOf('{');
    const lastBrace = raw.lastIndexOf('}');
    if (firstBrace !== -1 && lastBrace !== -1 && lastBrace >= firstBrace) {
      return JSON.parse(raw.slice(firstBrace, lastBrace + 1));
    }
    return JSON.parse(raw.trim());
  }

  // API 4: Interactive Custom Separation Execution
  app.post('/api/run-separation', (req, res) => {
    const config = req.body || {};
    const pythonProc = spawn('python3', [
      path.resolve(UNMIX_DIR, 'interactive_api.py'),
      'custom'
    ], {
      cwd: UNMIX_DIR,
      env: { ...process.env, PYTHONPATH: UNMIX_DIR }
    });

    let stdoutData = '';
    let stderrData = '';
    let isTerminated = false;

    // Timeout safety (30 seconds)
    const timer = setTimeout(() => {
      isTerminated = true;
      pythonProc.kill('SIGTERM');
      if (!res.headersSent) {
        res.status(504).json({ error: 'Separation timed out after 30 seconds.' });
      }
    }, 30000);

    try {
      pythonProc.stdin.write(JSON.stringify(config));
      pythonProc.stdin.end();
    } catch (writeErr) {
      clearTimeout(timer);
      return res.status(500).json({ error: 'Failed to write separation config to Python process' });
    }

    pythonProc.stdout.on('data', (chunk) => {
      stdoutData += chunk.toString();
    });

    pythonProc.stderr.on('data', (chunk) => {
      stderrData += chunk.toString();
    });

    pythonProc.on('close', (code) => {
      clearTimeout(timer);
      if (isTerminated || res.headersSent) return;

      if (code === 0) {
        try {
          const result = extractJson(stdoutData);
          res.json(result);
        } catch (e) {
          res.status(500).json({ error: 'Failed to parse separation result', raw: stdoutData });
        }
      } else {
        res.status(500).json({ error: 'Python separation failed', stderr: stderrData });
      }
    });

    pythonProc.on('error', (procErr) => {
      clearTimeout(timer);
      if (!res.headersSent) {
        res.status(500).json({ error: 'Failed to spawn Python process', details: procErr.message });
      }
    });
  });

  // API 4B: Upload and Unmix Audio Files
  app.post('/api/upload-and-unmix', (req, res) => {
    const config = req.body || {};
    const pythonProc = spawn('python3', [
      path.resolve(UNMIX_DIR, 'interactive_api.py'),
      'upload'
    ], {
      cwd: UNMIX_DIR,
      env: { ...process.env, PYTHONPATH: UNMIX_DIR }
    });

    let stdoutData = '';
    let stderrData = '';
    let isTerminated = false;

    // Timeout safety (45 seconds for user uploads)
    const timer = setTimeout(() => {
      isTerminated = true;
      pythonProc.kill('SIGTERM');
      if (!res.headersSent) {
        res.status(504).json({ error: 'Audio upload processing timed out after 45 seconds.' });
      }
    }, 45000);

    try {
      pythonProc.stdin.write(JSON.stringify(config));
      pythonProc.stdin.end();
    } catch (writeErr) {
      clearTimeout(timer);
      return res.status(500).json({ error: 'Failed to write upload data to Python process' });
    }

    pythonProc.stdout.on('data', (chunk) => {
      stdoutData += chunk.toString();
    });

    pythonProc.stderr.on('data', (chunk) => {
      stderrData += chunk.toString();
    });

    pythonProc.on('close', (code) => {
      clearTimeout(timer);
      if (isTerminated || res.headersSent) return;

      if (code === 0) {
        try {
          const result = extractJson(stdoutData);
          res.json(result);
        } catch (e) {
          res.status(500).json({ error: 'Failed to parse uploaded separation result', raw: stdoutData });
        }
      } else {
        res.status(500).json({ error: 'Python upload separation failed', stderr: stderrData });
      }
    });

    pythonProc.on('error', (procErr) => {
      clearTimeout(timer);
      if (!res.headersSent) {
        res.status(500).json({ error: 'Failed to spawn Python process', details: procErr.message });
      }
    });
  });

  // API 5: Mini-Shazam Fingerprint Comparison
  app.get('/api/mini-shazam', (req, res) => {
    const pythonProc = spawn('python3', [
      path.resolve(UNMIX_DIR, 'interactive_api.py'),
      'shazam'
    ], {
      cwd: UNMIX_DIR,
      env: { ...process.env, PYTHONPATH: UNMIX_DIR }
    });

    let stdoutData = '';
    let stderrData = '';

    pythonProc.stdout.on('data', (chunk) => {
      stdoutData += chunk.toString();
    });
    pythonProc.stderr.on('data', (chunk) => {
      stderrData += chunk.toString();
    });

    pythonProc.on('close', (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(stdoutData.trim());
          res.json(result);
        } catch (e) {
          res.status(500).json({ error: 'Failed to parse Shazam result', raw: stdoutData });
        }
      } else {
        res.status(500).json({ error: 'Shazam query failed', stderr: stderrData });
      }
    });
  });

  // API 6: Technical Documentation and Readme
  app.get('/api/readme', (req, res) => {
    const readmePath = path.resolve(UNMIX_DIR, 'README.md');
    if (fs.existsSync(readmePath)) {
      res.setHeader('Content-Type', 'text/markdown');
      res.send(fs.readFileSync(readmePath, 'utf-8'));
    } else {
      res.status(404).send('README not found');
    }
  });

  // Mount Vite middlewares in development
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.resolve(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Unmix the Signal (ICA) server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Fatal error starting server:', err);
  process.exit(1);
});
