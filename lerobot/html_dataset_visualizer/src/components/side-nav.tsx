"use client";

import Link from "next/link";
import React from "react";

interface SidebarProps {
  datasetInfo: any;
  paginatedEpisodes: any[];
  episodeId: any;
  totalPages: number;
  currentPage: number;
  prevPage: () => void;
  nextPage: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({
  datasetInfo,
  paginatedEpisodes,
  episodeId,
  totalPages,
  currentPage,
  prevPage,
  nextPage,
}) => {
  const [sidebarVisible, setSidebarVisible] = React.useState(true);
  const toggleSidebar = () => setSidebarVisible((prev) => !prev);

  const sidebarRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (!sidebarVisible) return;
    function handleClickOutside(event: MouseEvent) {
      // If click is outside the sidebar nav
      if (
        sidebarRef.current &&
        !sidebarRef.current.contains(event.target as Node)
      ) {
        setTimeout(() => setSidebarVisible(false), 500);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [sidebarVisible]);

  return (
    <div className="flex z-10 min-h-screen absolute md:static" ref={sidebarRef}>
      <nav
        className={`shrink-0 overflow-y-auto bg-slate-900 p-5 break-words md:max-h-screen w-60 md:shrink ${
          !sidebarVisible ? "hidden" : ""
        }`}
        aria-label="Sidebar navigation"
      >
        <ul>
          <li>Number of samples/frames: {datasetInfo.total_frames}</li>
          <li>Number of episodes: {datasetInfo.total_episodes}</li>
          <li>Frames per second: {datasetInfo.fps}</li>
        </ul>

        <p>Episodes:</p>

        {/* episodes menu for medium & large screens */}
        <div className="ml-2 block">
          <ul>
            {paginatedEpisodes.map((episode) => (
              <li key={episode} className="mt-0.5 font-mono text-sm">
                <Link
                  href={`./episode_${episode}`}
                  className={`underline ${episode === episodeId ? "-ml-1 font-bold" : ""}`}
                >
                  Episode {episode}
                </Link>
              </li>
            ))}
          </ul>

          {totalPages > 1 && (
            <div className="mt-3 flex items-center text-xs">
              <button
                onClick={prevPage}
                className={`mr-2 rounded bg-slate-800 px-2 py-1 ${
                  currentPage === 1 ? "cursor-not-allowed opacity-50" : ""
                }`}
                disabled={currentPage === 1}
              >
                « Prev
              </button>
              <span className="mr-2 font-mono">
                {currentPage} / {totalPages}
              </span>
              <button
                onClick={nextPage}
                className={`rounded bg-slate-800 px-2 py-1 ${
                  currentPage === totalPages
                    ? "cursor-not-allowed opacity-50"
                    : ""
                }`}
                disabled={currentPage === totalPages}
              >
                Next »
              </button>
            </div>
          )}
        </div>
      </nav>
      {/* Toggle sidebar button */}
      <button
        className="mx-1 flex items-center opacity-50 hover:opacity-100 focus:outline-none focus:ring-0"
        onClick={toggleSidebar}
        title="Toggle sidebar"
      >
        <div className="h-10 w-2 rounded-full bg-slate-500"></div>
      </button>
    </div>
  );
};

export default Sidebar;
