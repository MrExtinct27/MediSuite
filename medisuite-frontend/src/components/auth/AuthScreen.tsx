"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import type React from "react";
import { motion } from "framer-motion";
import { Lock, User as UserIcon, ArrowRight, AlertCircle } from "lucide-react";

import { api } from "@/lib/api";
import { useAuthStore } from "@/store/authStore";

const MIN_PASSWORD_LENGTH = 8;

/* ─── Cyberpunk field (matches the Submit Claim inputs) ─────────────── */
function CyberField({
  id,
  label,
  type,
  value,
  onChange,
  placeholder,
  icon,
  autoComplete,
}: {
  id: string;
  label: string;
  type: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  icon: React.ReactNode;
  autoComplete?: string;
}) {
  return (
    <div className="space-y-1.5">
      <label
        htmlFor={id}
        className="block text-[10px] uppercase tracking-widest text-[rgba(var(--ms-text-rgb),0.5)]"
        style={{ fontFamily: "var(--font-dm-mono)" }}
      >
        {label}
      </label>
      <div className="relative">
        <span className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-[rgba(var(--ms-accent-rgb),0.6)]">
          {icon}
        </span>
        <input
          id={id}
          type={type}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          autoComplete={autoComplete}
          className="w-full rounded-xl border border-[rgba(var(--ms-accent-rgb),0.12)] bg-[rgba(var(--ms-accent-rgb),0.04)] px-4 py-3 pl-10 text-sm text-[var(--ms-text)] placeholder-[rgba(var(--ms-text-rgb),0.25)] outline-none transition-all duration-200 focus:border-[var(--ms-accent)] focus:shadow-[0_0_0_1px_rgba(var(--ms-accent-rgb),0.3),0_0_12px_rgba(var(--ms-accent-rgb),0.1)]"
          style={{ fontFamily: "var(--font-dm-mono)" }}
        />
      </div>
    </div>
  );
}

export function AuthScreen({ mode }: { mode: "login" | "register" }) {
  const router = useRouter();
  const login = useAuthStore((s) => s.login);
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const hydrated = useAuthStore((s) => s.hydrated);

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const isRegister = mode === "register";
  const title = isRegister ? "Create Account" : "Welcome Back";
  const subtitle = isRegister
    ? "Register to start processing claims."
    : "Sign in to access your claims.";

  // Already signed in → skip the auth screens.
  useEffect(() => {
    if (hydrated && isAuthenticated) router.replace("/dashboard");
  }, [hydrated, isAuthenticated, router]);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      setError(null);

      const uname = username.trim();
      if (!uname) {
        setError("Username is required.");
        return;
      }
      if (isRegister && password.length < MIN_PASSWORD_LENGTH) {
        setError(`Password must be at least ${MIN_PASSWORD_LENGTH} characters.`);
        return;
      }
      if (!password) {
        setError("Password is required.");
        return;
      }

      setSubmitting(true);
      try {
        const res = await api.post<{ access_token: string; username: string }>(
          `/auth/${mode}`,
          { username: uname, password }
        );
        login(res.data.access_token, { username: res.data.username });
        router.replace("/dashboard");
      } catch (err) {
        const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
        setError(
          detail ||
            (isRegister
              ? "Registration failed. Please try again."
              : "Login failed. Please try again.")
        );
      } finally {
        setSubmitting(false);
      }
    },
    [username, password, isRegister, mode, login, router]
  );

  return (
    <div className="flex min-h-[calc(100vh-5rem)] items-center justify-center px-6 py-10">
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        className="glass-card w-full max-w-md p-8"
      >
        <div className="mb-6">
          <p
            className="mb-1 text-[10px] uppercase tracking-widest text-[var(--ms-accent)]"
            style={{ fontFamily: "var(--font-dm-mono)" }}
          >
            — {isRegister ? "Register" : "Login"}
          </p>
          <h1
            className="text-2xl font-bold uppercase text-[var(--ms-text)]"
            style={{ fontFamily: "var(--font-syne)", letterSpacing: "-0.02em" }}
          >
            {title}
          </h1>
          <p className="helper-text mt-2">{subtitle}</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <CyberField
            id="username"
            label="Username"
            type="text"
            value={username}
            onChange={setUsername}
            placeholder="your_username"
            autoComplete="username"
            icon={<UserIcon className="size-4" />}
          />
          <CyberField
            id="password"
            label="Password"
            type="password"
            value={password}
            onChange={setPassword}
            placeholder={isRegister ? "At least 8 characters" : "••••••••"}
            autoComplete={isRegister ? "new-password" : "current-password"}
            icon={<Lock className="size-4" />}
          />

          {error && (
            <motion.div
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-start gap-2 rounded-xl border border-[rgba(var(--ms-error-rgb),0.3)] bg-[rgba(var(--ms-error-rgb),0.08)] px-3 py-2.5"
            >
              <AlertCircle className="mt-0.5 size-3.5 shrink-0 text-[var(--ms-error)]" />
              <span
                className="text-xs text-[var(--ms-error)]"
                style={{ fontFamily: "var(--font-dm-sans)", lineHeight: 1.6 }}
              >
                {error}
              </span>
            </motion.div>
          )}

          <button
            type="submit"
            disabled={submitting}
            className="flex w-full items-center justify-center gap-2 rounded-2xl bg-[var(--ms-accent)] py-3.5 text-[11px] font-bold uppercase tracking-wider text-[var(--ms-on-accent)] transition-all hover:bg-[var(--ms-accent)]/85 hover:shadow-[0_0_20px_rgba(var(--ms-accent-rgb),0.35)] disabled:cursor-not-allowed disabled:opacity-50"
            style={{ fontFamily: "var(--font-syne)" }}
          >
            {submitting ? (
              <>
                <span className="size-4 animate-spin rounded-full border-2 border-[var(--ms-on-accent)]/30 border-t-[var(--ms-on-accent)]" />
                {isRegister ? "Creating…" : "Signing in…"}
              </>
            ) : (
              <>
                {isRegister ? "Create Account" : "Sign In"}
                <ArrowRight className="size-3.5" />
              </>
            )}
          </button>
        </form>

        <p className="helper-text mt-6 text-center">
          {isRegister ? "Already have an account? " : "Don't have an account? "}
          <Link
            href={isRegister ? "/login" : "/register"}
            className="font-semibold text-[var(--ms-accent)] underline-offset-2 hover:underline"
          >
            {isRegister ? "Sign in" : "Sign up"}
          </Link>
        </p>
      </motion.div>
    </div>
  );
}
