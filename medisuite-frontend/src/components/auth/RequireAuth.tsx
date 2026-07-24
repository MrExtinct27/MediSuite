"use client";

import { useRouter } from "next/navigation";
import { useEffect, type ReactNode } from "react";

import { useAuthStore } from "@/store/authStore";

/**
 * Wraps protected routes. Redirects unauthenticated users to /login.
 *
 * Renders a loader until the auth store has hydrated from localStorage, so there
 * is no flash of protected content before the redirect, and no hydration mismatch
 * (server and first client render both show the loader).
 */
export function RequireAuth({ children }: { children: ReactNode }) {
  const router = useRouter();
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const hydrated = useAuthStore((s) => s.hydrated);

  useEffect(() => {
    if (hydrated && !isAuthenticated) {
      router.replace("/login");
    }
  }, [hydrated, isAuthenticated, router]);

  if (!hydrated || !isAuthenticated) {
    return (
      <div className="flex min-h-[70vh] items-center justify-center">
        <div className="size-6 animate-spin rounded-full border-2 border-[rgba(var(--ms-accent-rgb),0.25)] border-t-[var(--ms-accent)]" />
      </div>
    );
  }

  return <>{children}</>;
}
