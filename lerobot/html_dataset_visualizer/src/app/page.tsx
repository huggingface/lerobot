"use client";
import { useEffect, useRef } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { redirect } from "next/navigation";

export default function Home({
  searchParams,
}: {
  searchParams: { [key: string]: string | undefined };
}) {
  // sync with hf.co/spaces URL params
  if (searchParams.path) {
    redirect(searchParams.path);
  }

  // leagcy sync with hf.co/spaces URL params
  let redirectUrl: string | null = null;
  if (searchParams?.dataset && searchParams?.episode) {
    redirectUrl = `/${searchParams.dataset}/episode_${searchParams.episode}`;
  } else if (searchParams?.dataset) {
    redirectUrl = `/${searchParams.dataset}`;
  }

  if (redirectUrl && searchParams?.t) {
    redirectUrl += `?t=${searchParams.t}`;
  }

  if (redirectUrl) {
    redirect(redirectUrl);
  }

  const playerRef = useRef<any>(null);

  useEffect(() => {
    // Load YouTube IFrame API if not already present
    if (!(window as any).YT) {
      const tag = document.createElement("script");
      tag.src = "https://www.youtube.com/iframe_api";
      document.body.appendChild(tag);
    }
    let interval: NodeJS.Timeout;
    (window as any).onYouTubeIframeAPIReady = () => {
      playerRef.current = new (window as any).YT.Player("yt-bg-player", {
        videoId: "Er8SPJsIYr0",
        playerVars: {
          autoplay: 1,
          mute: 1,
          controls: 0,
          showinfo: 0,
          modestbranding: 1,
          rel: 0,
          loop: 1,
          fs: 0,
          playlist: "Er8SPJsIYr0",
          start: 0,
        },
        events: {
          onReady: (event: any) => {
            event.target.playVideo();
            event.target.mute();
            interval = setInterval(() => {
              const t = event.target.getCurrentTime();
              if (t >= 60) {
                event.target.seekTo(0);
              }
            }, 500);
          },
        },
      });
    };
    return () => {
      if (interval) clearInterval(interval);
      if (playerRef.current && playerRef.current.destroy)
        playerRef.current.destroy();
    };
  }, []);

  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  const handleGo = (e: React.FormEvent) => {
    e.preventDefault();
    const value = inputRef.current?.value.trim();
    if (value) {
      router.push(value);
    }
  };

  return (
    <div className="relative h-screen w-screen overflow-hidden">
      {/* YouTube Video Background */}
      <div className="video-background">
        <div id="yt-bg-player" />
      </div>
      {/* Overlay */}
      <div className="fixed top-0 right-0 bottom-0 left-0 bg-black/60 -z-0" />
      {/* Centered Content */}
      <div className="relative z-10 h-screen flex flex-col items-center justify-center text-white text-center">
        <h1 className="text-4xl md:text-5xl font-bold mb-6 drop-shadow-lg">
          LeRobot Dataset Visualizer
        </h1>
        <a
          href="https://x.com/RemiCadene/status/1825455895561859185"
          target="_blank"
          rel="noopener noreferrer"
          className="text-sky-400 font-medium text-lg underline mb-8 inline-block hover:text-sky-300 transition-colors"
        >
          create & train your own robots
        </a>
        <form onSubmit={handleGo} className="flex gap-2 justify-center mt-6">
          <input
            ref={inputRef}
            type="text"
            placeholder="Enter dataset id (e.g. lerobot/visualize_dataset)"
            className="px-4 py-2 rounded-md text-base text-white border-white border-1 focus:outline-none min-w-[220px] shadow-md"
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                // Prevent double submission if form onSubmit also fires
                e.preventDefault();
                handleGo(e as any);
              }
            }}
          />
          <button
            type="submit"
            className="px-5 py-2 rounded-md bg-sky-400 text-black font-semibold text-base hover:bg-sky-300 transition-colors shadow-md"
          >
            Go
          </button>
        </form>
        {/* Example Datasets */}
        <div className="mt-8">
          <div className="font-semibold mb-2 text-lg">Example Datasets:</div>
          <div className="flex flex-col gap-2 items-center">
            {[
              "lerobot/aloha_static_cups_open",
              "lerobot/columbia_cairlab_pusht_real",
              "lerobot/taco_play",
            ].map((ds) => (
              <button
                key={ds}
                type="button"
                className="px-4 py-2 rounded bg-slate-700 text-sky-200 hover:bg-sky-700 hover:text-white transition-colors shadow"
                onClick={() => {
                  if (inputRef.current) {
                    inputRef.current.value = ds;
                    inputRef.current.focus();
                  }
                  router.push(ds);
                }}
              >
                {ds}
              </button>
            ))}
          </div>
        </div>

        <Link
          href="/explore"
          className="inline-block px-6 py-3 mt-8 rounded-md bg-sky-500 text-white font-semibold text-lg shadow-lg hover:bg-sky-400 transition-colors"
        >
          Explore Open Datasets
        </Link>
      </div>
    </div>
  );
}
