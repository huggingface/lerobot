import React from "react";
import { useTime } from "../context/time-context";
import {
  FaPlay,
  FaPause,
  FaBackward,
  FaForward,
  FaUndoAlt,
  FaArrowDown,
  FaArrowUp,
} from "react-icons/fa";

import { debounce } from "@/utils/debounce";

const PlaybackBar: React.FC = () => {
  const { duration, isPlaying, setIsPlaying, currentTime, setCurrentTime } =
    useTime();

  const sliderActiveRef = React.useRef(false);
  const wasPlayingRef = React.useRef(false);
  const [sliderValue, setSliderValue] = React.useState(currentTime);

  // Only update sliderValue from context if not dragging
  React.useEffect(() => {
    if (!sliderActiveRef.current) {
      setSliderValue(currentTime);
    }
  }, [currentTime]);

  const updateTime = debounce((t: number) => {
    setCurrentTime(t);
  }, 200);

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const t = Number(e.target.value);
    setSliderValue(t);
    updateTime(t);
  };

  const handleSliderMouseDown = () => {
    sliderActiveRef.current = true;
    wasPlayingRef.current = isPlaying;
    setIsPlaying(false);
  };

  const handleSliderMouseUp = () => {
    sliderActiveRef.current = false;
    setCurrentTime(sliderValue); // Snap to final value
    if (wasPlayingRef.current) {
      setIsPlaying(true);
    }
    // If it was paused before, keep it paused
  };

  return (
    <div className="flex items-center gap-4 w-full max-w-4xl mx-auto sticky bottom-0 bg-slate-900/95 px-4 py-3 rounded-3xl mt-auto">
      <button
        title="Jump backward 5 seconds"
        onClick={() => setCurrentTime(Math.max(0, currentTime - 5))}
        className="text-2xl hidden md:block"
      >
        <FaBackward size={24} />
      </button>
      <button
        className={`text-3xl transition-transform ${isPlaying ? "scale-90 opacity-60" : "scale-110"}`}
        title="Play. Toggle with Space"
        onClick={() => setIsPlaying(true)}
        style={{ display: isPlaying ? "none" : "inline-block" }}
      >
        <FaPlay size={24} />
      </button>
      <button
        className={`text-3xl transition-transform ${!isPlaying ? "scale-90 opacity-60" : "scale-110"}`}
        title="Pause. Toggle with Space"
        onClick={() => setIsPlaying(false)}
        style={{ display: !isPlaying ? "none" : "inline-block" }}
      >
        <FaPause size={24} />
      </button>
      <button
        title="Jump forward 5 seconds"
        onClick={() => setCurrentTime(Math.min(duration, currentTime + 5))}
        className="text-2xl hidden md:block"
      >
        <FaForward size={24} />
      </button>
      <button
        title="Rewind from start"
        onClick={() => setCurrentTime(0)}
        className="text-2xl hidden md:block"
      >
        <FaUndoAlt size={24} />
      </button>
      <input
        type="range"
        min={0}
        max={duration}
        step={0.01}
        value={sliderValue}
        onChange={handleSliderChange}
        onMouseDown={handleSliderMouseDown}
        onMouseUp={handleSliderMouseUp}
        onTouchStart={handleSliderMouseDown}
        onTouchEnd={handleSliderMouseUp}
        className="flex-1 mx-2 accent-orange-500 focus:outline-none focus:ring-0"
        aria-label="Seek video"
      />
      <span className="w-16 text-right tabular-nums text-xs text-slate-200 shrink-0">
        {Math.floor(sliderValue)} / {Math.floor(duration)}
      </span>

      <div className="text-xs text-slate-300 select-none ml-8 flex-col gap-y-0.5 hidden md:flex">
        <p>
          <span className="inline-flex items-center gap-1 font-mono align-middle">
            <span className="px-2 py-0.5 rounded border border-slate-400 bg-slate-800 text-slate-200 text-xs shadow-inner">
              Space
            </span>
          </span>{" "}
          to pause/unpause
        </p>
        <p>
          <span className="inline-flex items-center gap-1 font-mono align-middle">
            <FaArrowUp size={14} />/<FaArrowDown size={14} />
          </span>{" "}
          to previous/next episode
        </p>
      </div>
    </div>
  );
};

export default PlaybackBar;
