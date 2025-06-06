"use client";

import React, { useEffect, useRef } from "react";
import Link from "next/link";

import { useRouter, useSearchParams } from "next/navigation";
import { postParentMessageWithParams } from "@/utils/postParentMessage";

type ExploreGridProps = {
  datasets: Array<{ id: string; videoUrl: string | null }>;
  currentPage: number;
  totalPages: number;
};

export default function ExploreGrid({
  datasets,
  currentPage,
  totalPages,
}: ExploreGridProps) {
  // sync with parent window hf.co/spaces
  useEffect(() => {
    postParentMessageWithParams((params: URLSearchParams) => {
      params.set("path", window.location.pathname + window.location.search);
    });
  }, []);

  // Create an array of refs for each video
  const videoRefs = useRef<(HTMLVideoElement | null)[]>([]);

  return (
    <main className="p-8">
      <h1 className="text-2xl font-bold mb-6">Explore LeRobot Datasets</h1>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
        {datasets.map((ds, idx) => (
          <Link
            key={ds.id}
            href={`/${ds.id}`}
            className="relative border rounded-lg p-4 bg-white shadow hover:shadow-lg transition overflow-hidden h-48 flex items-end group"
            onMouseEnter={() => {
              const vid = videoRefs.current[idx];
              if (vid) vid.play();
            }}
            onMouseLeave={() => {
              const vid = videoRefs.current[idx];
              if (vid) {
                vid.pause();
                vid.currentTime = 0;
              }
            }}
          >
            <video
              ref={(el) => {
                videoRefs.current[idx] = el;
              }}
              src={ds.videoUrl || undefined}
              className="absolute top-0 left-0 w-full h-full object-cover object-center z-0"
              loop
              muted
              playsInline
              preload="metadata"
              onTimeUpdate={(e) => {
                const vid = e.currentTarget;
                if (vid.currentTime >= 15) {
                  vid.pause();
                  vid.currentTime = 0;
                }
              }}
            />
            <div className="absolute top-0 left-0 w-full h-full bg-black/40 z-10 pointer-events-none" />
            <div className="relative z-20 font-mono text-blue-100 break-all text-sm bg-black/60 backdrop-blur px-2 py-1 rounded shadow">
              {ds.id}
            </div>
          </Link>
        ))}
      </div>
      <div className="flex justify-center mt-8 gap-4">
        {currentPage > 1 && (
          <button
            className="px-6 py-2 bg-gray-600 text-white rounded shadow hover:bg-gray-700 transition"
            onClick={() => {
              const params = new URLSearchParams(window.location.search);
              params.set("p", (currentPage - 1).toString());
              window.location.search = params.toString();
            }}
          >
            Previous
          </button>
        )}
        {currentPage < totalPages && (
          <button
            className="px-6 py-2 bg-blue-600 text-white rounded shadow hover:bg-blue-700 transition"
            onClick={() => {
              const params = new URLSearchParams(window.location.search);
              params.set("p", (currentPage + 1).toString());
              window.location.search = params.toString();
            }}
          >
            Next
          </button>
        )}
      </div>
    </main>
  );
}
