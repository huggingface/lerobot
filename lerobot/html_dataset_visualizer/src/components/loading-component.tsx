"use client";

export default function Loading() {
  return (
    <div
      className="absolute inset-0 flex flex-col items-center justify-center bg-slate-950/70 z-10 text-slate-100 animate-fade-in"
      tabIndex={-1}
      aria-modal="true"
      role="dialog"
    >
      <svg
        className="animate-spin mb-8"
        width="64"
        height="64"
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <circle
          className="opacity-25"
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          strokeWidth="4"
        />
        <path
          className="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
        />
      </svg>
      <h1 className="text-2xl font-bold mb-2">Loading...</h1>
      <p className="text-slate-400">preparing data & videos</p>
    </div>
  );
}
