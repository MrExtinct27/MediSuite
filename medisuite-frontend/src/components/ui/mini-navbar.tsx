"use client";

import React, { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import axios from 'axios';
import { useTheme } from 'next-themes';
import { Sun, Moon, LogOut } from 'lucide-react';

import { useAuthStore } from '@/store/authStore';

const AnimatedNavLink = ({ href, children }: {
  href: string;
  children: React.ReactNode;
}) => {
  return (
    <Link
      href={href}
      className="whitespace-nowrap text-xs uppercase tracking-widest font-mono text-[rgba(var(--ms-text-rgb),0.55)] transition-all duration-200 hover:text-[var(--ms-accent)] hover:[text-shadow:0_0_8px_rgba(var(--ms-accent-rgb),0.8),0_0_16px_rgba(var(--ms-accent-rgb),0.4)]"
    >
      {children}
    </Link>
  );
};

export function Navbar() {
  const [isOpen, setIsOpen] = useState(false);
  const [apiStatus, setApiStatus] = useState<'connected' | 'offline' | 'checking'>('checking');
  const [headerShapeClass, setHeaderShapeClass] = useState('rounded-full');
  const shapeTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const pathname = usePathname();
  const router = useRouter();

  // Auth state — gate auth-dependent UI on `authHydrated` so SSR and the first
  // client render match (avoids a hydration mismatch / flash before the session is known).
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const authHydrated = useAuthStore((s) => s.hydrated);
  const authUser = useAuthStore((s) => s.user);
  const logout = useAuthStore((s) => s.logout);

  const handleLogout = () => {
    logout();
    router.push('/');
  };

  // Theme toggle — guard against hydration mismatch: next-themes resolves the
  // active theme on the client, so we render a same-sized placeholder until mount.
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  // Check API health on mount
  useEffect(() => {
    const checkApi = async () => {
      try {
        await axios.get(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/health`);
        setApiStatus('connected');
      } catch {
        setApiStatus('offline');
      }
    };
    checkApi();
    const interval = setInterval(checkApi, 30000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (shapeTimeoutRef.current) {
      clearTimeout(shapeTimeoutRef.current);
    }
    if (isOpen) {
      setHeaderShapeClass('rounded-2xl');
    } else {
      shapeTimeoutRef.current = setTimeout(() => {
        setHeaderShapeClass('rounded-full');
      }, 300);
    }
    return () => {
      if (shapeTimeoutRef.current) {
        clearTimeout(shapeTimeoutRef.current);
      }
    };
  }, [isOpen]);

  // Close mobile menu on route change
  useEffect(() => {
    setIsOpen(false);
  }, [pathname]);

  // Protected links only appear once we know the user is signed in.
  const navLinks = isAuthenticated
    ? [
        { label: 'Home', href: '/' },
        { label: 'Dashboard', href: '/dashboard' },
        { label: 'New Claim', href: '/claims/new' },
      ]
    : [{ label: 'Home', href: '/' }];

  const logoElement = (
    <Link href="/" className="flex items-center gap-2">
      <div className="relative w-6 h-6 flex items-center justify-center">
        <span className="absolute w-2 h-2 rounded-full bg-[var(--ms-accent)] top-0 left-1/2 -translate-x-1/2 opacity-90 shadow-[0_0_6px_var(--ms-accent)]" />
        <span className="absolute w-2 h-2 rounded-full bg-[var(--ms-accent)] left-0 top-1/2 -translate-y-1/2 opacity-90 shadow-[0_0_6px_var(--ms-accent)]" />
        <span className="absolute w-2 h-2 rounded-full bg-[var(--ms-accent)] right-0 top-1/2 -translate-y-1/2 opacity-90 shadow-[0_0_6px_var(--ms-accent)]" />
        <span className="absolute w-2 h-2 rounded-full bg-[var(--ms-accent)] bottom-0 left-1/2 -translate-x-1/2 opacity-90 shadow-[0_0_6px_var(--ms-accent)]" />
        <span className="w-1.5 h-1.5 rounded-full bg-[var(--ms-accent)] opacity-60" />
      </div>
      <span className="font-mono text-sm font-bold tracking-widest uppercase">
        <span className="text-[var(--ms-text)]">Medi</span>
        <span className="text-[var(--ms-accent)]">Suite</span>
      </span>
    </Link>
  );

  const apiStatusElement = (
    <div className="flex items-center gap-2">
      <div className="relative flex items-center gap-1.5">
        <span
          className={`w-2 h-2 rounded-full ${
            apiStatus === 'connected'
              ? 'bg-[var(--ms-success)] shadow-[0_0_6px_var(--ms-success)]'
              : apiStatus === 'offline'
                ? 'bg-[var(--ms-error)] shadow-[0_0_6px_var(--ms-error)]'
                : 'bg-[var(--ms-warning)]'
          } ${apiStatus === 'connected' ? 'animate-pulse' : ''}`}
        />
        <span className="font-mono text-[10px] uppercase tracking-widest text-[rgba(var(--ms-text-rgb),0.55)] hidden sm:block">
          {apiStatus === 'connected'
            ? 'API Connected'
            : apiStatus === 'offline'
              ? 'API Offline'
              : 'Checking...'}
        </span>
      </div>
    </div>
  );

  // Auth controls: username + Logout when signed in, Login + Sign Up otherwise.
  // Rendered as a fixed-size placeholder until hydrated to avoid a layout flash.
  const authControls = !authHydrated ? (
    <div className="h-8 w-28" aria-hidden />
  ) : isAuthenticated ? (
    <div className="flex items-center gap-3">
      <span className="hidden md:inline font-mono text-xs uppercase tracking-widest text-[rgba(var(--ms-text-rgb),0.75)] whitespace-nowrap">
        {authUser?.username}
      </span>
      <button
        type="button"
        onClick={handleLogout}
        className="flex items-center gap-1.5 px-4 py-1.5 text-xs font-mono font-semibold uppercase tracking-widest text-[var(--ms-error)] border border-[rgba(var(--ms-error-rgb),0.4)] bg-[rgba(var(--ms-error-rgb),0.08)] rounded-full hover:bg-[rgba(var(--ms-error-rgb),0.16)] transition-all duration-200 whitespace-nowrap"
      >
        <LogOut className="size-3.5" /> Logout
      </button>
    </div>
  ) : (
    <div className="flex items-center gap-3">
      <Link
        href="/login"
        className="font-mono text-xs font-semibold uppercase tracking-widest text-[rgba(var(--ms-text-rgb),0.75)] hover:text-[var(--ms-accent)] transition-colors duration-200 whitespace-nowrap"
      >
        Login
      </Link>
      <Link
        href="/register"
        className="px-4 py-1.5 text-xs font-mono font-semibold uppercase tracking-widest text-[var(--ms-on-accent)] bg-[var(--ms-accent)] rounded-full hover:brightness-110 transition-all duration-200 whitespace-nowrap"
      >
        Sign Up
      </Link>
    </div>
  );

  // Sun/Moon theme toggle. Mount-guarded: render a same-sized placeholder until
  // the client resolves the theme, avoiding a hydration mismatch.
  const themeToggle = mounted ? (
    <button
      type="button"
      onClick={() => setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')}
      aria-label={resolvedTheme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
      title={resolvedTheme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
      className="flex size-8 items-center justify-center rounded-full border border-[rgba(var(--ms-accent-rgb),0.2)] bg-[rgba(var(--ms-accent-rgb),0.06)] text-[rgba(var(--ms-text-rgb),0.7)] transition-all duration-200 hover:border-[var(--ms-accent)] hover:text-[var(--ms-accent)]"
    >
      {resolvedTheme === 'dark' ? <Sun className="size-4" /> : <Moon className="size-4" />}
    </button>
  ) : (
    <div className="size-8 rounded-full border border-[rgba(var(--ms-accent-rgb),0.2)] bg-[rgba(var(--ms-accent-rgb),0.06)]" aria-hidden />
  );

  return (
    <header
      className={`fixed top-4 left-1/2 -translate-x-1/2 z-50 flex flex-col items-center px-6 py-3 backdrop-blur-md border border-[rgba(var(--ms-accent-rgb),0.15)] bg-[rgba(var(--ms-bg-rgb),0.8)] w-[calc(100%-2rem)] sm:w-auto ${headerShapeClass} transition-[border-radius] duration-300`}
    >
      <div className="flex items-center justify-between w-full gap-x-8">
        {logoElement}

        <nav className="hidden sm:flex items-center space-x-6">
          {navLinks.map((link) => (
            <AnimatedNavLink key={link.href} href={link.href}>
              {link.label}
            </AnimatedNavLink>
          ))}
        </nav>

        <div className="hidden sm:flex items-center gap-4">
          {apiStatusElement}
          {themeToggle}
          {authControls}
        </div>

        <button
          className="sm:hidden text-[rgba(var(--ms-text-rgb),0.7)] hover:text-[var(--ms-accent)] transition-colors"
          onClick={() => setIsOpen(!isOpen)}
          aria-label={isOpen ? 'Close Menu' : 'Open Menu'}
        >
          {isOpen ? (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          ) : (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          )}
        </button>
      </div>

      {/* Mobile Menu */}
      <div
        className={`sm:hidden flex flex-col items-center w-full transition-all duration-300 overflow-hidden ${
          isOpen ? 'max-h-96 opacity-100 pt-4' : 'max-h-0 opacity-0 pt-0 pointer-events-none'
        }`}
      >
        <nav className="flex flex-col items-center space-y-4 w-full">
          {navLinks.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="font-mono text-xs uppercase tracking-widest text-[rgba(var(--ms-text-rgb),0.55)] hover:text-[var(--ms-accent)] transition-colors w-full text-center"
            >
              {link.label}
            </Link>
          ))}
        </nav>
        <div className="mt-4 w-full flex flex-col items-center gap-3">
          {apiStatusElement}
          {themeToggle}
          {authControls}
        </div>
      </div>
    </header>
  );
}
