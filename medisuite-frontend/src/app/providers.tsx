"use client";

import { ThemeProvider } from "next-themes";
import { useEffect } from "react";
import type { ReactNode } from "react";

import { useAuthStore } from "@/store/authStore";

/**
 * Client-side providers. Wraps the app in next-themes so the dark/light choice
 * persists across navigation and reloads. Dark stays the default to match the
 * existing design; system preference is disabled.
 *
 * Also restores the auth session from localStorage once on mount so the token
 * survives reloads and the rest of the app can rely on the store being hydrated.
 */
export function Providers({ children }: { children: ReactNode }) {
  useEffect(() => {
    useAuthStore.getState().hydrate();
  }, []);

  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="dark"
      enableSystem={false}
      disableTransitionOnChange
    >
      {children}
    </ThemeProvider>
  );
}
