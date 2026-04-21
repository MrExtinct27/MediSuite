"use client";

import { motion } from "framer-motion";
import Link from "next/link";

function FloatingPaths({ position }: { position: number }) {
  const paths = Array.from({ length: 36 }, (_, i) => ({
    id: i,
    d: `M-${380 - i * 5 * position} -${189 + i * 6}C-${
      380 - i * 5 * position
    } -${189 + i * 6} -${312 - i * 5 * position} ${216 - i * 6} ${
      152 - i * 5 * position
    } ${343 - i * 6}C${616 - i * 5 * position} ${470 - i * 6} ${
      684 - i * 5 * position
    } ${875 - i * 6} ${684 - i * 5 * position} ${875 - i * 6}`,
    width: 0.5 + i * 0.03,
  }));

  return (
    <div className="absolute inset-0 pointer-events-none">
      <svg className="w-full h-full" viewBox="0 0 696 316" fill="none">
        <title>Medical Data Flow Paths</title>
        {paths.map((path) => (
          <motion.path
            key={path.id}
            d={path.d}
            stroke="#00D4FF"
            strokeWidth={path.width}
            strokeOpacity={0.05 + path.id * 0.02}
            initial={{ pathLength: 0.3, opacity: 0.4 }}
            animate={{
              pathLength: 1,
              opacity: [0.2, 0.5, 0.2],
              pathOffset: [0, 1, 0],
            }}
            transition={{
              duration: 20 + (path.id % 10),
              repeat: Number.POSITIVE_INFINITY,
              ease: "linear",
            }}
          />
        ))}
      </svg>
    </div>
  );
}

export function HeroSection() {
  const titleWords = ["AGENTIC AI For", "MEDICAL CLAIMS"];

  return (
    <div className="relative min-h-screen w-full flex items-center justify-center overflow-hidden bg-[#050D1A]">
      {/* Grid background */}
      <div
        className="absolute inset-0"
        style={{
          backgroundImage: `
            linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px)`,
          backgroundSize: "40px 40px",
        }}
      />

      {/* Animated paths */}
      <div className="absolute inset-0">
        <FloatingPaths position={1} />
        <FloatingPaths position={-1} />
      </div>

      {/* Radial glow */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: `
            radial-gradient(ellipse at 30% 50%, rgba(0,212,255,0.06) 0%, transparent 60%),
            radial-gradient(ellipse at 70% 50%, rgba(0,255,156,0.04) 0%, transparent 60%)`,
        }}
      />

      {/* Content */}
      <div className="relative z-10 container mx-auto px-4 md:px-6 text-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 2 }}
          className="max-w-5xl mx-auto"
        >
          {/* Label */}
          <motion.div
            initial={{ opacity: 0, y: -12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="mb-8"
          >
            <span
              className="glass-card-sm inline-flex items-center gap-2 px-4 py-1.5 text-[10px] uppercase tracking-widest text-[rgba(228,240,255,0.6)]"
              style={{ fontFamily: "var(--font-dm-mono)" }}
            >
              <span
                className="neon-pulse size-1.5 rounded-full bg-[#00FF9C]"
                style={{ color: "#00FF9C" }}
              />
              Clinical Intelligence Platform
            </span>
          </motion.div>

          {/* Main heading with letter animation */}
          <h1
            className="font-black mb-6 tracking-tight uppercase leading-none"
            style={{
              fontFamily: "var(--font-space-grotesk)",
              fontSize: "clamp(3.2rem, 8vw, 6.5rem)",
              letterSpacing: "-0.04em",
            }}
          >
            {titleWords.map((word, wordIndex) => (
              <span key={wordIndex} className="block mb-2">
                {word.split("").map((letter, letterIndex) => (
                  <motion.span
                    key={`${wordIndex}-${letterIndex}`}
                    initial={{ y: 100, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{
                      delay: wordIndex * 0.15 + letterIndex * 0.03,
                      type: "spring",
                      stiffness: 150,
                      damping: 25,
                    }}
                    className={`inline-block ${letter === " " ? "mr-4" : ""} text-transparent bg-clip-text bg-gradient-to-b ${
                      wordIndex === 2
                        ? "from-[#00D4FF] to-[#00D4FF]/70"
                        : "from-white to-white/70"
                    }`}
                  >
                    {letter === " " ? "\u00A0" : letter}
                  </motion.span>
                ))}
              </span>
            ))}
          </h1>

          {/* Subheading */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.2, duration: 0.8 }}
            className="text-lg md:text-xl text-gray-400 max-w-2xl mx-auto mb-10 leading-relaxed"
            style={{ fontFamily: "var(--font-dm-sans)", lineHeight: 1.7 }}
          >
            Four specialized AI agents that automate ICD-10 coding, validation,
            and CMS-1500 generation with full explainability trails
          </motion.p>

          {/* CTA Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.5, duration: 0.8 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            {/* Primary CTA */}
            <div className="inline-block group relative bg-gradient-to-b from-[#00D4FF]/20 to-[#00D4FF]/5 p-px rounded-2xl backdrop-blur-lg overflow-hidden shadow-lg hover:shadow-[0_0_30px_rgba(0,212,255,0.3)] transition-shadow duration-300">
              <Link
                href="/claims/new"
                className="flex items-center gap-2 rounded-[1.15rem] px-8 py-3.5 text-sm font-mono font-semibold uppercase tracking-widest backdrop-blur-md bg-[#00D4FF] hover:bg-white text-black transition-all duration-300 group-hover:-translate-y-0.5 border border-[#00D4FF]/20"
                style={{ fontFamily: "var(--font-dm-mono)" }}
              >
                <span className="opacity-90 group-hover:opacity-100 transition-opacity">
                  Process Claim
                </span>
                <span className="opacity-70 group-hover:opacity-100 group-hover:translate-x-1.5 transition-all duration-300">
                  →
                </span>
              </Link>
            </div>

            {/* Secondary CTA */}
            <div className="inline-block group relative bg-gradient-to-b from-white/5 to-white/[0.02] p-px rounded-2xl backdrop-blur-lg overflow-hidden shadow-lg hover:shadow-[0_0_20px_rgba(0,212,255,0.15)] transition-shadow duration-300">
              <Link
                href="/dashboard"
                className="flex items-center gap-2 rounded-[1.15rem] px-8 py-3.5 text-sm font-mono font-semibold uppercase tracking-widest backdrop-blur-md bg-[rgba(5,13,26,0.95)] hover:bg-[rgba(0,212,255,0.1)] text-[#00D4FF] transition-all duration-300 group-hover:-translate-y-0.5 border border-[#00D4FF]/20"
                style={{ fontFamily: "var(--font-dm-mono)" }}
              >
                <span className="opacity-90 group-hover:opacity-100">
                  View Dashboard
                </span>
                <span className="opacity-70 group-hover:opacity-100 group-hover:translate-x-1.5 transition-all duration-300">
                  →
                </span>
              </Link>
            </div>
          </motion.div>

          {/* Floating stats cards */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.8, duration: 0.8 }}
            className="flex flex-wrap justify-center gap-4 mt-16"
          >
            {[
              { label: "Avg Confidence", value: "94.8%", accent: "#00D4FF" },
              { label: "Processing Time", value: "15.4s", accent: "#00FF9C" },
              { label: "Cost Per Claim", value: "$0.026", accent: "#00D4FF" },
              { label: "ICD-10 Codes", value: "70,000+", accent: "#00FF9C" },
            ].map((stat, i) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 2.0 + i * 0.1 }}
                className="px-6 py-3 rounded-2xl backdrop-blur-md bg-[rgba(0,212,255,0.025)] border border-[rgba(0,212,255,0.08)]"
                style={{ borderTop: `1px solid ${stat.accent}40` }}
              >
                <div
                  className="text-xl font-bold"
                  style={{ fontFamily: "var(--font-dm-mono)", color: stat.accent }}
                >
                  {stat.value}
                </div>
                <div
                  className="text-xs uppercase tracking-widest text-gray-500 mt-0.5"
                  style={{ fontFamily: "var(--font-dm-mono)" }}
                >
                  {stat.label}
                </div>
              </motion.div>
            ))}
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
}
