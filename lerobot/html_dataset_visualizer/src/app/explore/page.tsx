import React from "react";
import ExploreGrid from "./explore-grid";
import {
  DatasetMetadata,
  fetchJson,
  formatStringWithVars,
} from "@/utils/parquetUtils";

export default async function ExplorePage({
  searchParams,
}: {
  searchParams: { p?: string };
}) {
  let datasets: any[] = [];
  let currentPage = 1;
  let totalPages = 1;
  try {
    const res = await fetch(
      "https://huggingface.co/api/datasets?sort=lastModified&filter=LeRobot",
      {
        cache: "no-store",
      },
    );
    if (!res.ok) throw new Error("Failed to fetch datasets");
    const data = await res.json();
    const allDatasets = data.datasets || data;
    // Use searchParams from props
    const page = parseInt(searchParams?.p || "1", 10);
    const perPage = 30;

    currentPage = page;
    totalPages = Math.ceil(allDatasets.length / perPage);

    const startIdx = (currentPage - 1) * perPage;
    const endIdx = startIdx + perPage;
    datasets = allDatasets.slice(startIdx, endIdx);
  } catch (e) {
    return <div className="p-8 text-red-600">Failed to load datasets.</div>;
  }

  // Fetch episode 0 data for each dataset
  const datasetWithVideos = (
    await Promise.all(
      datasets.map(async (ds: any) => {
        try {
          const [org, dataset] = ds.id.split("/");
          const repoId = `${org}/${dataset}`;
          const jsonUrl = `https://huggingface.co/datasets/${repoId}/resolve/main/meta/info.json`;
          const info = await fetchJson<DatasetMetadata>(jsonUrl);
          const videoEntry = Object.entries(info.features).find(
            ([key, value]) => value.dtype === "video",
          );
          let videoUrl: string | null = null;
          if (videoEntry) {
            const [key] = videoEntry;
            const videoPath = formatStringWithVars(info.video_path, {
              video_key: key,
              episode_chunk: "0".padStart(3, "0"),
              episode_index: "0".padStart(6, "0"),
            });
            const url =
              `https://huggingface.co/datasets/${repoId}/resolve/main/` +
              videoPath;
            // Check if videoUrl exists (status 200)
            try {
              const headRes = await fetch(url, { method: "HEAD" });
              if (headRes.ok) {
                videoUrl = url;
              }
            } catch (e) {
              // If fetch fails, videoUrl remains null
            }
          }
          return videoUrl ? { id: repoId, videoUrl } : null;
        } catch (err) {
          console.error(
            `Failed to fetch or parse dataset info for ${ds.id}:`,
            err,
          );
          return null;
        }
      }),
    )
  ).filter(Boolean) as { id: string; videoUrl: string | null }[];

  return (
    <ExploreGrid
      datasets={datasetWithVideos}
      currentPage={currentPage}
      totalPages={totalPages}
    />
  );
}
