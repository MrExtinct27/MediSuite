"use client";

import React, { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import axios from 'axios';

const AnimatedNavLink = ({ href, children }: {
  href: string;
  children: React.ReactNode;
}) => {
  return (
    <Link
      href={href}
      className="whitespace-nowrap text-xs uppercase tracking-widest font-mono text-gray-400 transition-all duration-200 hover:text-[#00D4FF] hover:[text-shadow:0_0_8px_rgba(0,212,255,0.8),0_0_16px_rgba(0,212,255,0.4)]"
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

  const navLinks = [
    { label: 'Home', href: '/' },
    { label: 'Dashboard', href: '/dashboard' },
    { label: 'New Claim', href: '/claims/new' },
  ];

  const logoElement = (
    <Link href="/" className="flex items-center gap-2">
      <div className="relative w-6 h-6 flex items-center justify-center">
        <span className="absolute w-2 h-2 rounded-full bg-[#00D4FF] top-0 left-1/2 -translate-x-1/2 opacity-90 shadow-[0_0_6px_#00D4FF]" />
        <span className="absolute w-2 h-2 rounded-full bg-[#00D4FF] left-0 top-1/2 -translate-y-1/2 opacity-90 shadow-[0_0_6px_#00D4FF]" />
        <span className="absolute w-2 h-2 rounded-full bg-[#00D4FF] right-0 top-1/2 -translate-y-1/2 opacity-90 shadow-[0_0_6px_#00D4FF]" />
        <span className="absolute w-2 h-2 rounded-full bg-[#00D4FF] bottom-0 left-1/2 -translate-x-1/2 opacity-90 shadow-[0_0_6px_#00D4FF]" />
        <span className="w-1.5 h-1.5 rounded-full bg-[#00D4FF] opacity-60" />
      </div>
      <span className="font-mono text-sm font-bold tracking-widest uppercase">
        <span className="text-white">Medi</span>
        <span className="text-[#00D4FF]">Suite</span>
      </span>
    </Link>
  );

  const apiStatusElement = (
    <div className="flex items-center gap-2">
      <div className="relative flex items-center gap-1.5">
        <span
          className={`w-2 h-2 rounded-full ${
            apiStatus === 'connected'
              ? 'bg-[#00FF9C] shadow-[0_0_6px_#00FF9C]'
              : apiStatus === 'offline'
                ? 'bg-[#FF3B6B] shadow-[0_0_6px_#FF3B6B]'
                : 'bg-yellow-400'
          } ${apiStatus === 'connected' ? 'animate-pulse' : ''}`}
        />
        <span className="font-mono text-[10px] uppercase tracking-widest text-gray-400 hidden sm:block">
          {apiStatus === 'connected'
            ? 'API Connected'
            : apiStatus === 'offline'
              ? 'API Offline'
              : 'Checking...'}
        </span>
      </div>
    </div>
  );

  const ctaButton = (
    <div className="relative group">
      <div className="absolute inset-0 -m-1 rounded-full bg-[#00D4FF] opacity-20 blur-lg pointer-events-none transition-all duration-300 group-hover:opacity-40 group-hover:blur-xl" />
      <Link
        href="/claims/new"
        className="relative z-10 px-4 py-1.5 text-xs font-mono font-semibold uppercase tracking-widest text-black bg-[#00D4FF] rounded-full hover:bg-white transition-all duration-200 whitespace-nowrap"
      >
        Process Claim
      </Link>
    </div>
  );

  return (
    <header
      className={`fixed top-4 left-1/2 -translate-x-1/2 z-50 flex flex-col items-center px-6 py-3 backdrop-blur-md border border-[rgba(0,212,255,0.15)] bg-[rgba(5,13,26,0.8)] w-[calc(100%-2rem)] sm:w-auto ${headerShapeClass} transition-[border-radius] duration-300`}
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
          {ctaButton}
        </div>

        <button
          className="sm:hidden text-gray-300 hover:text-[#00D4FF] transition-colors"
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
              className="font-mono text-xs uppercase tracking-widest text-gray-400 hover:text-[#00D4FF] transition-colors w-full text-center"
            >
              {link.label}
            </Link>
          ))}
        </nav>
        <div className="mt-4 w-full flex flex-col items-center gap-3">
          {apiStatusElement}
          {ctaButton}
        </div>
      </div>
    </header>
  );
}
