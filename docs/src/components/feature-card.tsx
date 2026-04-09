// ABOUTME: Feature cards with Lucide icons for the docs landing page.
// ABOUTME: Maps icon names to Lucide components for use in MDX files.

import { Card, Cards } from 'fumadocs-ui/components/card';
import {
  Layers,
  Puzzle,
  BarChart3,
  Database,
} from 'lucide-react';
import type { ReactNode } from 'react';
import { cn } from '@/lib/cn';

const icons: Record<string, ReactNode> = {
  layers: <Layers />,
  puzzle: <Puzzle />,
  'bar-chart': <BarChart3 />,
  database: <Database />,
};

const accentClasses: Record<string, string> = {
  layers:
    'border-sky-300/70 bg-sky-50/75 dark:border-sky-700/70 dark:bg-sky-950/30',
  puzzle:
    'border-sky-300/70 bg-sky-50/75 dark:border-sky-700/70 dark:bg-sky-950/30',
  'bar-chart':
    'border-sky-300/70 bg-sky-50/75 dark:border-sky-700/70 dark:bg-sky-950/30',
  database:
    'border-sky-300/70 bg-sky-50/75 dark:border-sky-700/70 dark:bg-sky-950/30',
};

export function FeatureCard({
  icon,
  title,
  description,
}: {
  icon: string;
  title: string;
  description: string;
}) {
  return (
    <Card
      icon={icons[icon]}
      title={title}
      description={description}
      className={cn(accentClasses[icon], 'shadow-sm')}
    />
  );
}

export { Cards as FeatureCards };
