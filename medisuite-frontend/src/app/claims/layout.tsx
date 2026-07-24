"use client";

import type { ReactNode } from "react";

import { RequireAuth } from "@/components/auth/RequireAuth";

// Protects every /claims/* route (new claim + claim detail).
export default function ClaimsLayout({ children }: { children: ReactNode }) {
  return <RequireAuth>{children}</RequireAuth>;
}
