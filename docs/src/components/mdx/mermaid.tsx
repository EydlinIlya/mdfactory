'use client';

import { useTheme } from 'next-themes';
import mermaid from 'mermaid';
import { useEffect, useId, useRef, useState } from 'react';

interface MermaidProps {
  chart: string;
}

export function Mermaid({ chart }: MermaidProps): React.ReactElement {
  const id = useId();
  const ref = useRef<HTMLDivElement>(null);
  const { resolvedTheme } = useTheme();
  const [svg, setSvg] = useState<string | null>(null);

  useEffect(() => {
    if (!ref.current) return;

    mermaid.initialize({
      startOnLoad: false,
      theme: resolvedTheme === 'dark' ? 'dark' : 'default',
    });

    void mermaid
      .render(id, chart)
      .then((result) => {
        setSvg(result.svg);
        return result;
      })
      .catch((error) => {
        console.error('Mermaid rendering failed:', error);
      });
  }, [chart, id, resolvedTheme]);

  return (
    <div
      ref={ref}
      className="flex flex-col items-center"
      dangerouslySetInnerHTML={svg ? { __html: svg } : undefined}
    />
  );
}
