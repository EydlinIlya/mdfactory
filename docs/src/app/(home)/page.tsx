'use client';

import Link from 'next/link';
import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function HomePage() {
  const router = useRouter();

  useEffect(() => {
    router.replace('/docs/');
  }, [router]);

  return (
    <div className="flex min-h-screen items-center justify-center px-6 text-center">
      <p className="text-sm text-neutral-600">
        Redirecting to documentation. If nothing happens,{' '}
        <Link className="underline underline-offset-4" href="/docs/">
          open the docs
        </Link>
        .
      </p>
    </div>
  );
}
