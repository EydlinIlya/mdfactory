# Agent Notes for docs/

## Scope
This file applies to the `docs/` folder and overrides the root `AGENTS.md` for documentation-related tasks.

## Overview
This is a [Fumadocs](https://fumadocs.dev) + Next.js documentation site for mdfactory. It combines hand-written guides with auto-generated API reference pages from Python docstrings.

## Tech Stack
- **Runtime/Package Manager**: Bun (not npm/yarn)
- **Framework**: Next.js 16 with static export
- **Docs Framework**: Fumadocs (fumadocs-ui, fumadocs-core, fumadocs-mdx)
- **Python API Docs**: fumadocs-python + fumapy-generate

## Key Commands
```bash
bun install                # Install dependencies
bun run docs:generate      # Generate API docs from Python
bun run dev                # Dev server (runs docs:generate first)
bun run build              # Production build (runs docs:generate first)
bun run types:check        # TypeScript type checking
```

## Directory Structure
```
docs/
├── content/docs/          # Hand-written MDX documentation
│   └── api/               # AUTO-GENERATED - do not edit manually
├── mdfactory-api/         # AUTO-GENERATED - intermediate JSON from fumapy
├── scripts/
│   └── generate-docs.mjs  # Orchestrates API doc generation
├── src/
│   ├── components/
│   │   └── python/        # Custom Python doc components (see below)
│   ├── app/               # Next.js app router pages
│   └── mdx-components.tsx # MDX component registry
└── source.config.ts       # Fumadocs configuration
```

## Auto-Generated Content (gitignored)
These are regenerated on every build - never edit manually:
- `content/docs/api/` - MDX files for API reference
- `mdfactory-api/` - JSON output from fumapy-generate
- `src/components/python/type-map.json` - Type-to-URL mapping

## Custom Components (Important!)

### Why Custom Python Components?
The default `fumadocs-python/components` render type annotations as plain text. We created custom wrappers in `src/components/python/` to add **automatic linking** to mdfactory classes and functions.

### How It Works
1. `scripts/generate-docs.mjs` runs `fumapy-generate` to extract Python docstrings
2. It also calls `buildTypeMap()` which scans the JSON and generates `type-map.json` mapping class/function names to their doc URLs
3. `src/components/python/type-link.tsx` imports this map and parses type annotations to create clickable links
4. `src/components/python/index.tsx` wraps the fumadocs-python components (`PyFunction`, `PyAttribute`, `PyParameter`, `PyFunctionReturn`) to use `TypeLink`
5. `src/mdx-components.tsx` imports from `@/components/python` instead of `fumadocs-python/components`

### What Gets Linked
- **Attribute types**: `type={"BuildInput"}` → clickable link to BuildInput docs
- **Parameter types**: `type={"SingleMoleculeSpecies"}` → clickable link
- **Function return types**: `(self) -> SystemComposition` → return type is linked
- **Fully qualified paths**: `mdfactory.models.input.BuildInput` → normalized and linked

### Adding New Linkable Types
No manual work needed! The type map is auto-generated from `mdfactory.json` during `bun run docs:generate`. Any new classes or functions in mdfactory will automatically be linkable.

## GitHub Actions
The `.github/workflows/docs.yml` workflow:
1. Installs Python 3.11 and Bun
2. Installs mdfactory and fumadocs-python
3. Runs `bun run build` (which generates API docs + builds static site)
4. Deploys `docs/out/` to GitHub Pages

## Common Tasks

### Adding a New Guide
Create an MDX file in `content/docs/` with frontmatter:
```mdx
---
title: My Guide
description: What this guide covers
---

Content here...
```

### Updating API Docs
Just run `bun run docs:generate` - it re-extracts from the Python source.

### Modifying Type Linking Behavior
Edit `src/components/python/type-link.tsx` for parsing logic or `src/components/python/index.tsx` for component rendering.

## Troubleshooting

### "Cannot find module 'type-map.json'"
Run `bun run docs:generate` first - the type map is generated at build time.

### API docs not updating
The `content/docs/api/` folder is regenerated fresh each time. If changes aren't appearing, ensure mdfactory is installed (`pip install -e ..`) and run `bun run docs:generate`.

### Type links not working for a class
Check that the class is exported and has a docstring. The type map only includes items found in `mdfactory.json`.
