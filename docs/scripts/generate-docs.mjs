import { rimraf } from 'rimraf';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import * as Python from 'fumadocs-python';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const apiDir = path.resolve(__dirname, '..', 'mdfactory-api');
const jsonPath = path.join(apiDir, 'mdfactory.json');
const typeMapPath = path.resolve(__dirname, '..', 'src/components/python/type-map.json');
const docsContentDir = path.resolve(__dirname, '..', 'content/docs');
const generatedApiDir = path.join(docsContentDir, 'api');

/**
 * Extract all classes and functions from the module tree to build a type map.
 * Maps short names to their doc URLs.
 */
function buildTypeMap(mod, baseUrl = '/docs/api') {
  const map = {};
  
  function processModule(module, currentPath) {
    // Process classes
    for (const [name, cls] of Object.entries(module.classes || {})) {
      const docPath = cls.path.replace(/^mdfactory\./, '').replaceAll('.', '/');
      map[name] = `${baseUrl}/${docPath}`;
    }
    
    // Process functions at module level
    for (const [name, func] of Object.entries(module.functions || {})) {
      // Store function paths for potential linking (less common but possible)
      const docPath = func.path.replace(/^mdfactory\./, '').replaceAll('.', '/');
      // Functions are rendered inline in module pages, link to module with anchor
      const modulePath = module.path.replace(/^mdfactory\./, '').replaceAll('.', '/');
      map[`${name}()`] = `${baseUrl}/${modulePath}#${name}`;
    }
    
    // Recurse into submodules
    for (const submod of Object.values(module.modules || {})) {
      processModule(submod, currentPath);
    }
  }
  
  processModule(mod, '');
  return map;
}

function run(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: 'inherit', ...options });
    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`${command} ${args.join(' ')} exited with code ${code}`));
      }
    });
  });
}

async function generate() {
  await fs.mkdir(apiDir, { recursive: true });
  await run('fumapy-generate', ['mdfactory', '--dir', apiDir], {
    cwd: path.resolve(__dirname, '..', '..'),
  });

  const outDir = generatedApiDir;

  await rimraf(outDir, { preserveRoot: false });

  const raw = await fs.readFile(jsonPath, 'utf8');
  const content = JSON.parse(raw);

  const converted = Python.convert(content, {
    baseUrl: '/docs/api',
    modulePrefix: 'mdfactory',
  });

  await Python.write(converted, {
    outDir,
    transformers: [
      (page) => {
        // Filter out __all__ attribute
        if (page.type === 'module' && page.data.attributes) {
          page.data.attributes = page.data.attributes.filter(
            attr => attr.name !== '__all__'
          );
        }
        
        return page;
      }
    ]
  });
  
  // Post-process all generated MDX files to fix URLs
  async function fixUrls(dir) {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      
      if (entry.isDirectory()) {
        await fixUrls(fullPath);
      } else if (entry.name.endsWith('.mdx')) {
        let content = await fs.readFile(fullPath, 'utf8');
        content = content.replace(/href=\{"\/docs\/api\/mdfactory\//g, 'href={"/docs/api/');
        await fs.writeFile(fullPath, content, 'utf8');
      }
    }
  }
  
  await fixUrls(outDir);
  
  // Generate type map for auto-linking
  const typeMap = buildTypeMap(content);
  await fs.writeFile(typeMapPath, JSON.stringify(typeMap, null, 2));
  console.log(`Generated type map with ${Object.keys(typeMap).length} entries`);
}

async function writeFallbackTypeMap() {
  await fs.mkdir(path.dirname(typeMapPath), { recursive: true });
  await fs.writeFile(typeMapPath, '{}\n', 'utf8');
}

async function writeFallbackApiDocs() {
  await fs.mkdir(generatedApiDir, { recursive: true });

  const meta = {
    title: 'API Reference',
    root: true,
    pages: ['index'],
  };

  const indexMdx = `---
title: API Reference
description: Generated Python API docs for mdfactory
---

<Callout type="warning">
  The generated Python API reference is not available in this build.
</Callout>

The docs site builds the API section from the local Python package with \`bun run docs:generate\`.

To restore the full API reference and clickable Python type links:

\`\`\`bash
cd docs
bun run docs:generate
\`\`\`

If generation still fails, ensure:

- the local package is installed with \`pip install -e ..\`
- \`fumapy-generate\` is available from the docs environment
`;

  await fs.writeFile(path.join(generatedApiDir, 'meta.json'), `${JSON.stringify(meta, null, 2)}\n`, 'utf8');
  await fs.writeFile(path.join(generatedApiDir, 'index.mdx'), indexMdx, 'utf8');
}

/**
 * Copy example YAML files from repo root to docs content directory.
 * This allows fumadocs <include> to access them without path traversal.
 */
async function copyExamples() {
  const repoRoot = path.resolve(__dirname, '..', '..');
  const examplesSource = path.join(repoRoot, 'examples');
  const examplesDest = path.resolve(__dirname, '..', 'content/examples');

  // Clean and recreate destination
  await rimraf(examplesDest, { preserveRoot: false });
  await fs.mkdir(examplesDest, { recursive: true });

  // Copy specific example directories
  const dirs = ['mixedbox', 'bilayer', 'lnp'];
  for (const dir of dirs) {
    const srcDir = path.join(examplesSource, dir);
    const destDir = path.join(examplesDest, dir);

    try {
      await fs.mkdir(destDir, { recursive: true });
      const files = await fs.readdir(srcDir);

      for (const file of files) {
        if (file.endsWith('.yaml') || file.endsWith('.yml')) {
          await fs.copyFile(path.join(srcDir, file), path.join(destDir, file));
        }
      }
      console.log(`Copied examples/${dir}/ YAML files`);
    } catch (err) {
      console.warn(`Warning: Could not copy examples/${dir}/: ${err.message}`);
    }
  }
}

// Run both tasks, but don't fail everything if API generation fails
// (e.g., when fumapy-generate is not installed)
async function main() {
  // Always copy examples first
  await copyExamples();

  // Try to generate API docs
  try {
    await generate();
  } catch (error) {
    await writeFallbackTypeMap();
    await writeFallbackApiDocs();
    console.warn('Warning: API doc generation failed (fumapy-generate may not be installed)');
    console.warn(error.message);

    // Ensure type-map.json exists even when generation fails
    try {
      await fs.access(typeMapPath);
    } catch {
      await fs.mkdir(path.dirname(typeMapPath), { recursive: true });
      await fs.writeFile(typeMapPath, '{}');
      console.log('Created empty type-map.json fallback');
    }
  }
}

main().catch((error) => {
  console.error('Failed to generate docs');
  console.error(error);
  process.exit(1);
});
