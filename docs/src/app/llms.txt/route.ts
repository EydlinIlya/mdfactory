// ABOUTME: Generates /llms.txt index listing all doc pages with descriptions.
// ABOUTME: Follows the llms.txt spec (https://llmstxt.org) for AI-agent discovery.

import { source } from '@/lib/source';

export const revalidate = false;

export async function GET() {
  const pages = source.getPages();

  const entries = pages
    .map((page) => `- [${page.data.title}](${page.url}): ${page.data.description ?? ''}`)
    .join('\n');

  const content = `# MDFactory

> High-throughput molecular dynamics simulation library for building, parametrizing, and analyzing GROMACS workflows.

## Docs

${entries}
`;

  return new Response(content);
}
