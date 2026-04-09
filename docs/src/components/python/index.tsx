import { cva } from 'class-variance-authority';
import { twMerge as cn } from 'tailwind-merge';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from 'fumadocs-ui/components/ui/collapsible';
import { buttonVariants } from 'fumadocs-ui/components/ui/button';
import { ChevronRight } from 'lucide-react';
import { highlight } from 'fumadocs-core/highlight';
import { TypeLink } from './type-link';
import type { ReactNode } from 'react';

const cardVariants = cva('bg-fd-card rounded-lg text-sm my-6 p-3 border');
const badgeVariants = cva(
  'text-xs font-medium border p-1 rounded-lg not-prose',
  {
    variants: {
      color: {
        func: 'bg-fdpy-func/10 text-fdpy-func border-fdpy-func/50',
        attribute: 'bg-fdpy-attribute/10 text-fdpy-attribute border-fdpy-attribute/50',
        class: 'bg-fdpy-class/10 text-fdpy-class border-fdpy-class/50',
        primary: 'bg-fd-primary/10 text-fd-primary border-fd-primary/10',
      },
    },
  }
);

/**
 * Renders a function signature with linked return type.
 * Parses signatures like "(self) -> BuildInput" or "(inp: BuildInput) -> None"
 */
function FunctionSignature({ signature }: { signature: string }) {
  // Split on " -> " to separate params from return type
  const arrowIndex = signature.lastIndexOf(' -> ');
  
  if (arrowIndex === -1) {
    // No return type annotation, just render as-is
    return <span className="text-fd-muted-foreground">{signature}</span>;
  }
  
  const params = signature.slice(0, arrowIndex);
  const returnType = signature.slice(arrowIndex + 4); // Skip " -> "
  
  return (
    <span className="text-fd-muted-foreground">
      {params} → <TypeLink annotation={returnType} />
    </span>
  );
}

interface PyFunctionProps {
  name: string;
  type: string;
  children?: ReactNode;
}

export function PyFunction({ name, type, children }: PyFunctionProps) {
  return (
    <figure className={cn(cardVariants())}>
      <div className="flex gap-2 items-center font-mono flex-wrap mb-4">
        <code className={cn(badgeVariants({ color: 'func' }))}>func</code>
        {name}
        <span className="not-prose text-xs">
          <FunctionSignature signature={type} />
        </span>
      </div>
      <div className="text-fd-muted-foreground prose-no-margin">{children}</div>
    </figure>
  );
}

interface PyAttributeProps {
  name: string;
  type?: string | null;
  value?: string | null;
  children?: ReactNode;
}

export function PyAttribute({ name, type, value, children }: PyAttributeProps) {
  return (
    <figure className={cn(cardVariants())}>
      <div className="flex gap-2 items-center flex-wrap font-mono mb-4">
        <code className={cn(badgeVariants({ color: 'attribute' }))}>attribute</code>
        {name}
        {type && (
          <span className="not-prose text-fd-muted-foreground text-xs">
            <TypeLink annotation={type} />
          </span>
        )}
      </div>
      <div className="text-fd-muted-foreground prose-no-margin">
        {value && (
          <InlineCode
            lang="python"
            className="not-prose text-xs"
            code={`= ${value}`}
          />
        )}
        {children}
      </div>
    </figure>
  );
}

interface PyParameterProps {
  name: string;
  type?: string | null;
  value?: string | null;
  children?: ReactNode;
}

export function PyParameter({ name, type, value, children }: PyParameterProps) {
  return (
    <div
      data-parameter=""
      className="bg-fd-secondary rounded-lg text-sm p-3 border shadow-md rounded-none first:rounded-t-lg last:rounded-b-lg"
    >
      <div className="flex flex-wrap gap-2 items-center font-mono text-fd-foreground">
        <code className={cn(badgeVariants({ color: 'primary' }))}>param</code>
        {name}
        {type && (
          <span className="ms-auto text-fd-muted-foreground not-prose text-xs">
            <TypeLink annotation={type} />
          </span>
        )}
      </div>
      <div className="text-fd-muted-foreground prose-no-margin mt-4 empty:hidden">
        {value ? (
          <InlineCode lang="python" code={`= ${value}`} className="not-prose text-xs" />
        ) : null}
        {children}
      </div>
    </div>
  );
}

interface PySourceCodeProps {
  children?: ReactNode;
}

export function PySourceCode({ children }: PySourceCodeProps) {
  return (
    <Collapsible className="my-6">
      <CollapsibleTrigger
        className={cn(
          buttonVariants({
            color: 'secondary',
            size: 'sm',
            className: 'group',
          })
        )}
      >
        Source Code
        <ChevronRight className="size-3.5 text-fd-muted-foreground group-data-[state=open]:rotate-90" />
      </CollapsibleTrigger>
      <CollapsibleContent className="prose-no-margin">{children}</CollapsibleContent>
    </Collapsible>
  );
}

interface PyFunctionReturnProps {
  type?: string | null;
  children?: ReactNode;
}

export function PyFunctionReturn({ type, children }: PyFunctionReturnProps) {
  return (
    <div className="border bg-fd-secondary rounded-lg p-3 mt-2">
      <div className="flex flex-wrap gap-2 not-prose">
        <p className="font-medium me-auto">Returns</p>
        <span className="text-xs">
          <TypeLink annotation={type ?? 'None'} />
        </span>
      </div>
      {children}
    </div>
  );
}

interface InlineCodeProps {
  lang: string;
  code: string;
  className?: string;
}

async function InlineCode({ lang, code, ...rest }: InlineCodeProps) {
  return highlight(code, {
    lang,
    components: {
      pre: (props) => (
        <span {...props} {...rest} className={cn(rest.className, props.className)} />
      ),
    },
  });
}

// Re-export Tabs from fumadocs-ui for completeness
export { Tab, Tabs } from 'fumadocs-ui/components/tabs';
