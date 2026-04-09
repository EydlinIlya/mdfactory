import defaultMdxComponents from 'fumadocs-ui/mdx';
import * as Python from '@/components/python';
import * as TabsComponents from 'fumadocs-ui/components/tabs';
import { Step, Steps } from 'fumadocs-ui/components/steps';
import { File, Folder, Files } from 'fumadocs-ui/components/files';
import { Accordion, Accordions } from 'fumadocs-ui/components/accordion';
import { Mermaid } from '@/components/mdx/mermaid';
import { FeatureCard, FeatureCards } from '@/components/feature-card';
import type { MDXComponents } from 'mdx/types';

export function getMDXComponents(components?: MDXComponents): MDXComponents {
  return {
    ...defaultMdxComponents,
    ...Python,
    ...TabsComponents,
    Step,
    Steps,
    File,
    Folder,
    Files,
    Accordion,
    Accordions,
    Mermaid,
    FeatureCard,
    FeatureCards,
    ...components,
  };
}
