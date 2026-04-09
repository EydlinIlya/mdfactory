import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

const explicitBasePath = process.env.NEXT_PUBLIC_BASE_PATH;
const normalizedBasePath =
  explicitBasePath === '__root__' ? '' : explicitBasePath;
const basePath = normalizedBasePath !== undefined ? normalizedBasePath : '';

/** @type {import('next').NextConfig} */
const config = {
  reactStrictMode: true,
  output: 'export',
  trailingSlash: true,
  basePath: basePath || undefined,
  assetPrefix: basePath || undefined,
  env: {
    NEXT_PUBLIC_BASE_PATH: basePath,
  },
  images: {
    unoptimized: true,
  },
};

export default withMDX(config);
