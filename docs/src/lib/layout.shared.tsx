import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';

export function baseOptions(): BaseLayoutProps {
  const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';
  const logoSrc = `${basePath}/mdfactory-logo-v1.0.png`;

  return {
    nav: {
      title: (
        <span className="inline-flex items-center gap-2">
          <img src={logoSrc} alt="MDFactory" className="h-6 w-auto" />
          <span>MDFactory</span>
        </span>
      ),
    },
    // links: [
    //   {
    //     text: 'Documentation',
    //     url: '/docs',
    //   },
    //   {
    //     text: 'API Reference',
    //     url: '/docs/api',
    //   },
    // ],
  };
}
