"use client";

import { useEffect, useRef, useState } from "react";
import { useTime } from "../context/time-context";
import { FaExpand, FaCompress, FaTimes, FaEye } from "react-icons/fa";

type VideoInfo = {
  filename: string;
  url: string;
};

type VideoPlayerProps = {
  videosInfo: VideoInfo[];
  onVideosReady?: () => void;
};

export const VideosPlayer = ({
  videosInfo,
  onVideosReady,
}: VideoPlayerProps) => {
  const { currentTime, setCurrentTime, isPlaying, setIsPlaying } = useTime();
  const videoRefs = useRef<HTMLVideoElement[]>([]);
  // Hidden/enlarged state and hidden menu
  const [hiddenVideos, setHiddenVideos] = useState<string[]>([]);
  // Find the index of the first visible (not hidden) video
  const firstVisibleIdx = videosInfo.findIndex(
    (video) => !hiddenVideos.includes(video.filename),
  );
  // Count of visible videos
  const visibleCount = videosInfo.filter(
    (video) => !hiddenVideos.includes(video.filename),
  ).length;
  const [enlargedVideo, setEnlargedVideo] = useState<string | null>(null);
  // Track previous hiddenVideos for comparison
  const prevHiddenVideosRef = useRef<string[]>([]);
  const videoContainerRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const [showHiddenMenu, setShowHiddenMenu] = useState(false);
  const hiddenMenuRef = useRef<HTMLDivElement | null>(null);
  const showHiddenBtnRef = useRef<HTMLButtonElement | null>(null);
  const [videoCodecError, setVideoCodecError] = useState(false);

  // Initialize video refs
  useEffect(() => {
    videoRefs.current = videoRefs.current.slice(0, videosInfo.length);
  }, [videosInfo]);

  // When videos get unhidden, start playing them if it was playing
  useEffect(() => {
    // Find which videos were just unhidden
    const prevHidden = prevHiddenVideosRef.current;
    const newlyUnhidden = prevHidden.filter(
      (filename) => !hiddenVideos.includes(filename),
    );
    if (newlyUnhidden.length > 0) {
      videosInfo.forEach((video, idx) => {
        if (newlyUnhidden.includes(video.filename)) {
          const ref = videoRefs.current[idx];
          if (ref) {
            ref.currentTime = currentTime;
            if (isPlaying) {
              ref.play().catch(() => {});
            }
          }
        }
      });
    }
    prevHiddenVideosRef.current = hiddenVideos;
  }, [hiddenVideos, isPlaying, videosInfo, currentTime]);

  // Check video codec support
  useEffect(() => {
    const checkCodecSupport = () => {
      const dummyVideo = document.createElement("video");
      const canPlayVideos = dummyVideo.canPlayType(
        'video/mp4; codecs="av01.0.05M.08"',
      );
      setVideoCodecError(!canPlayVideos);
    };

    checkCodecSupport();
  }, []);

  // Handle play/pause
  useEffect(() => {
    videoRefs.current.forEach((video) => {
      if (video) {
        if (isPlaying) {
          video.play().catch((e) => console.error("Error playing video:", e));
        } else {
          video.pause();
        }
      }
    });
  }, [isPlaying]);

  // Minimize enlarged video on Escape key
  useEffect(() => {
    if (!enlargedVideo) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setEnlargedVideo(null);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    // Scroll enlarged video into view
    const ref = videoContainerRefs.current[enlargedVideo];
    if (ref) {
      ref.scrollIntoView();
    }
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [enlargedVideo]);

  // Close hidden videos dropdown on outside click
  useEffect(() => {
    if (!showHiddenMenu) return;
    function handleClick(e: MouseEvent) {
      const menu = hiddenMenuRef.current;
      const btn = showHiddenBtnRef.current;
      if (
        menu &&
        !menu.contains(e.target as Node) &&
        btn &&
        !btn.contains(e.target as Node)
      ) {
        setShowHiddenMenu(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [showHiddenMenu]);

  // Close dropdown if no hidden videos
  useEffect(() => {
    if (hiddenVideos.length === 0 && showHiddenMenu) {
      setShowHiddenMenu(false);
    }
    // Minimize if enlarged video is hidden
    if (enlargedVideo && hiddenVideos.includes(enlargedVideo)) {
      setEnlargedVideo(null);
    }
  }, [hiddenVideos, showHiddenMenu, enlargedVideo]);

  // Sync video times
  useEffect(() => {
    videoRefs.current.forEach((video) => {
      if (video && Math.abs(video.currentTime - currentTime) > 0.2) {
        video.currentTime = currentTime;
      }
    });
  }, [currentTime]);

  // Handle time update
  const handleTimeUpdate = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    const video = e.target as HTMLVideoElement;
    if (video && video.duration) {
      setCurrentTime(video.currentTime);
    }
  };

  // Handle video ready
  useEffect(() => {
    let videosReadyCount = 0;
    const onCanPlayThrough = () => {
      videosReadyCount += 1;
      if (videosReadyCount === videosInfo.length) {
        // Start autoplay when all videos are ready
        if (typeof onVideosReady === "function") {
          onVideosReady();
          setIsPlaying(true);
        }
      }
    };
    videoRefs.current.forEach((video) => {
      if (video) {
        video.addEventListener("canplaythrough", onCanPlayThrough);
      }
    });
    return () => {
      videoRefs.current.forEach((video) => {
        if (video) {
          video.removeEventListener("canplaythrough", onCanPlayThrough);
        }
      });
    };
  }, []);

  return (
    <>
      {/* Error message */}
      {videoCodecError && (
        <div className="font-medium text-orange-700">
          <p>
            Videos could NOT play because{" "}
            <a
              href="https://en.wikipedia.org/wiki/AV1"
              target="_blank"
              className="underline"
            >
              AV1
            </a>{" "}
            decoding is not available on your browser.
          </p>
          <ul className="list-inside list-decimal">
            <li>
              If iPhone:{" "}
              <span className="italic">
                It is supported with A17 chip or higher.
              </span>
            </li>
            <li>
              If Mac with Safari:{" "}
              <span className="italic">
                It is supported on most browsers except Safari with M1 chip or
                higher and on Safari with M3 chip or higher.
              </span>
            </li>
            <li>
              Other:{" "}
              <span className="italic">
                Contact the maintainers on LeRobot discord channel:
              </span>
              <a
                href="https://discord.com/invite/s3KuuzsPFb"
                target="_blank"
                className="underline"
              >
                https://discord.com/invite/s3KuuzsPFb
              </a>
            </li>
          </ul>
        </div>
      )}

      {/* Show Hidden Videos Button */}
      {hiddenVideos.length > 0 && (
        <div className="relative">
          <button
            ref={showHiddenBtnRef}
            className="flex items-center gap-2 rounded bg-slate-800 px-3 py-2 text-sm text-slate-100 hover:bg-slate-700 border border-slate-500"
            onClick={() => setShowHiddenMenu((prev) => !prev)}
          >
            <FaEye /> Show Hidden Videos ({hiddenVideos.length})
          </button>
          {showHiddenMenu && (
            <div
              ref={hiddenMenuRef}
              className="absolute left-0 mt-2 w-max rounded border border-slate-500 bg-slate-900 shadow-lg p-2 z-50"
            >
              <div className="mb-2 text-xs text-slate-300">
                Restore hidden videos:
              </div>
              {hiddenVideos.map((filename) => (
                <button
                  key={filename}
                  className="block w-full text-left px-2 py-1 rounded hover:bg-slate-700 text-slate-100"
                  onClick={() =>
                    setHiddenVideos((prev: string[]) =>
                      prev.filter((v: string) => v !== filename),
                    )
                  }
                >
                  {filename}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Videos */}
      <div className="flex flex-wrap gap-x-2 gap-y-6">
        {videosInfo.map((video, idx) => {
          if (hiddenVideos.includes(video.filename) || videoCodecError)
            return null;
          const isEnlarged = enlargedVideo === video.filename;
          return (
            <div
              key={video.filename}
              ref={(el) => {
                videoContainerRefs.current[video.filename] = el;
              }}
              className={`${isEnlarged ? "z-40 fixed inset-0 bg-black bg-opacity-90 flex flex-col items-center justify-center" : "max-w-96"}`}
              style={isEnlarged ? { height: "100vh", width: "100vw" } : {}}
            >
              <p className="truncate w-full rounded-t-xl bg-gray-800 px-2 text-sm text-gray-300 flex items-center justify-between">
                <span>{video.filename}</span>
                <span className="flex gap-1">
                  <button
                    title={isEnlarged ? "Minimize" : "Enlarge"}
                    className="ml-2 p-1 hover:bg-slate-700 rounded focus:outline-none focus:ring-0"
                    onClick={() =>
                      setEnlargedVideo(isEnlarged ? null : video.filename)
                    }
                  >
                    {isEnlarged ? <FaCompress /> : <FaExpand />}
                  </button>
                  <button
                    title="Hide Video"
                    className="ml-1 p-1 hover:bg-slate-700 rounded focus:outline-none focus:ring-0"
                    onClick={() =>
                      setHiddenVideos((prev: string[]) => [
                        ...prev,
                        video.filename,
                      ])
                    }
                    disabled={visibleCount === 1}
                  >
                    <FaTimes />
                  </button>
                </span>
              </p>
              <video
                ref={(el) => {
                  if (el) videoRefs.current[idx] = el;
                }}
                muted
                loop
                className={`w-full object-contain ${isEnlarged ? "max-h-[90vh] max-w-[90vw]" : ""}`}
                onTimeUpdate={
                  idx === firstVisibleIdx ? handleTimeUpdate : undefined
                }
                style={isEnlarged ? { zIndex: 41 } : {}}
              >
                <source src={video.url} type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            </div>
          );
        })}
      </div>
    </>
  );
};

export default VideosPlayer;
