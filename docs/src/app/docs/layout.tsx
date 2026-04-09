import { source } from '@/lib/source';
import { DocsLayout } from 'fumadocs-ui/layouts/docs';
import { baseOptions } from '@/lib/layout.shared';

export default function Layout({ children }: LayoutProps<'/docs'>) {
  return (
    <DocsLayout
      tree={source.pageTree}
      sidebar={{
        footer: (
          <p className="text-xs text-fd-muted-foreground">
            API docs regenerate from the Python package on every build.
          </p>
        ),
      }}
      {...baseOptions()}
    >
      {children}
    </DocsLayout>
  );
}
