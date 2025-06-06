import React, {
  createContext,
  useContext,
  useRef,
  useState,
  useCallback,
} from "react";

// The shape of our context
type TimeContextType = {
  currentTime: number;
  setCurrentTime: (t: number) => void;
  subscribe: (cb: (t: number) => void) => () => void;
  isPlaying: boolean;
  setIsPlaying: React.Dispatch<React.SetStateAction<boolean>>;
  duration: number;
  setDuration: React.Dispatch<React.SetStateAction<number>>;
};

const TimeContext = createContext<TimeContextType | undefined>(undefined);

export const useTime = () => {
  const ctx = useContext(TimeContext);
  if (!ctx) throw new Error("useTime must be used within a TimeProvider");
  return ctx;
};

export const TimeProvider: React.FC<{
  children: React.ReactNode;
  duration: number;
}> = ({ children, duration: initialDuration }) => {
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(initialDuration);
  const listeners = useRef<Set<(t: number) => void>>(new Set());

  // Call this to update time and notify all listeners
  const updateTime = useCallback((t: number) => {
    setCurrentTime(t);
    listeners.current.forEach((fn) => fn(t));
  }, []);

  // Components can subscribe to time changes (for imperative updates)
  const subscribe = useCallback((cb: (t: number) => void) => {
    listeners.current.add(cb);
    return () => listeners.current.delete(cb);
  }, []);

  return (
    <TimeContext.Provider
      value={{
        currentTime,
        setCurrentTime: updateTime,
        subscribe,
        isPlaying,
        setIsPlaying,
        duration,
        setDuration,
      }}
    >
      {children}
    </TimeContext.Provider>
  );
};
