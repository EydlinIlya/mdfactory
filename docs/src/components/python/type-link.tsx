'use client';

import Link from 'next/link';
import { Fragment, type ReactNode } from 'react';
import typeMap from './type-map.json';

// Type map is auto-generated from mdfactory.json during docs:generate
const TYPE_MAP: Record<string, string> = typeMap;

/**
 * Parse a type annotation string and return an array of parts,
 * where mdfactory class names are converted to link objects.
 */
function parseTypeAnnotation(annotation: string): Array<{ type: 'text' | 'link'; value: string; href?: string }> {
  const parts: Array<{ type: 'text' | 'link'; value: string; href?: string }> = [];
  
  // Get class names (entries without parentheses are classes)
  const classNames = Object.keys(TYPE_MAP).filter(k => !k.endsWith('()'));
  
  if (classNames.length === 0) {
    return [{ type: 'text', value: annotation }];
  }
  
  // First, replace fully qualified mdfactory paths with short names
  const normalizedAnnotation = annotation.replace(
    /mdfactory\.[\w.]+\.(\w+)/g, 
    (_, className) => className
  );
  
  // Create pattern for class names (only match word boundaries)
  const shortNamePattern = new RegExp(`\\b(${classNames.join('|')})\\b`, 'g');
  
  // Find all class name occurrences
  const matches: Array<{ index: number; match: string; className: string }> = [];
  let match: RegExpExecArray | null;
  
  while ((match = shortNamePattern.exec(normalizedAnnotation)) !== null) {
    const className = match[1];
    if (TYPE_MAP[className]) {
      matches.push({
        index: match.index,
        match: match[0],
        className,
      });
    }
  }
  
  // Build parts array
  let currentIndex = 0;
  for (const m of matches) {
    // Add text before this match
    if (m.index > currentIndex) {
      parts.push({
        type: 'text',
        value: normalizedAnnotation.slice(currentIndex, m.index),
      });
    }
    // Add the link
    parts.push({
      type: 'link',
      value: m.className,
      href: TYPE_MAP[m.className],
    });
    currentIndex = m.index + m.match.length;
  }
  
  // Add remaining text
  if (currentIndex < normalizedAnnotation.length) {
    parts.push({
      type: 'text',
      value: normalizedAnnotation.slice(currentIndex),
    });
  }
  
  // If no matches, return the original annotation as text
  if (parts.length === 0) {
    parts.push({ type: 'text', value: annotation });
  }
  
  return parts;
}

interface TypeLinkProps {
  annotation: string | null;
  className?: string;
}

/**
 * Renders a type annotation with links to mdfactory internal classes.
 */
export function TypeLink({ annotation, className }: TypeLinkProps): ReactNode {
  if (!annotation) return null;
  
  const parts = parseTypeAnnotation(annotation);
  
  return (
    <span className={className}>
      {parts.map((part, i) => (
        <Fragment key={i}>
          {part.type === 'link' ? (
            <Link 
              href={part.href!}
              className="text-fd-primary hover:underline"
            >
              {part.value}
            </Link>
          ) : (
            part.value
          )}
        </Fragment>
      ))}
    </span>
  );
}
