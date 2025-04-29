"use client";

import React from "react";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="flex h-screen items-center justify-center bg-slate-950 text-red-400">
      <div className="max-w-xl p-8 rounded bg-slate-900 border border-red-500 shadow-lg">
        <h2 className="text-2xl font-bold mb-4">Something went wrong</h2>
        <p className="text-lg font-mono whitespace-pre-wrap mb-4">
          {error.message}
        </p>
        <button
          className="mt-4 px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
          onClick={() => reset()}
        >
          Try Again
        </button>
      </div>
    </div>
  );
}
