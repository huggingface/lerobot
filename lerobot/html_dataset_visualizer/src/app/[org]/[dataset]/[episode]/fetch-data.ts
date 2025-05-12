import {
  DatasetMetadata,
  fetchJson,
  fetchParquetFile,
  formatStringWithVars,
  readParquetColumn,
} from "@/utils/parquetUtils";
import { pick } from "@/utils/pick";

const DATASET_URL =
  process.env.DATASET_URL || "https://huggingface.co/datasets";

const SERIES_NAME_DELIMITER = " | ";

export async function getEpisodeData(
  org: string,
  dataset: string,
  episodeId: number,
) {
  const repoId = `${org}/${dataset}`;
  try {
    const episode_chunk = Math.floor(0 / 1000);
    const jsonUrl = `${DATASET_URL}/${repoId}/resolve/main/meta/info.json`;

    const info = await fetchJson<DatasetMetadata>(jsonUrl);

    // Dataset information
    const datasetInfo = {
      repoId,
      total_frames: info.total_frames,
      total_episodes: info.total_episodes,
      fps: info.fps,
    };

    // Generate list of episodes
    const episodes = Array.from(
      { length: datasetInfo.total_episodes },
      // episode id starts from 1
      (_, i) => i + 1,
    );

    // Videos information
    const videosInfo = Object.entries(info.features)
      .filter(([key, value]) => value.dtype === "video")
      .map(([key, _]) => {
        const videoPath = formatStringWithVars(info.video_path, {
          video_key: key,
          episode_chunk: episode_chunk.toString().padStart(3, "0"),
          episode_index: episodeId.toString().padStart(6, "0"),
        });
        return {
          filename: key,
          url: `${DATASET_URL}/${repoId}/resolve/main/` + videoPath,
        };
      });

    // Column data
    const columnNames = Object.entries(info.features)
      .filter(
        ([key, value]) =>
          ["float32", "int32"].includes(value.dtype) &&
          value.shape.length === 1,
      )
      .map(([key, { shape }]) => ({ key, length: shape[0] }));

    // Exclude specific columns
    const excludedColumns = [
      "timestamp",
      "frame_index",
      "episode_index",
      "index",
      "task_index",
    ];
    const filteredColumns = columnNames.filter(
      (column) => !excludedColumns.includes(column.key),
    );
    const filteredColumnNames = [
      "timestamp",
      ...filteredColumns.map((column) => column.key),
    ];

    const columns = filteredColumns.map(({ key }) => {
      let column_names = info.features[key].names;
      while (typeof column_names === "object") {
        if (Array.isArray(column_names)) break;
        column_names = Object.values(column_names ?? {})[0];
      }
      return {
        key,
        value: Array.isArray(column_names)
          ? column_names.map((name) => `${key}${SERIES_NAME_DELIMITER}${name}`)
          : Array.from(
              { length: columnNames.find((c) => c.key === key)?.length ?? 1 },
              (_, i) => `${key}${SERIES_NAME_DELIMITER}${i}`,
            ),
      };
    });

    const parquetUrl =
      `${DATASET_URL}/${repoId}/resolve/main/` +
      formatStringWithVars(info.data_path, {
        episode_chunk: episode_chunk.toString().padStart(3, "0"),
        episode_index: episodeId.toString().padStart(6, "0"),
      });

    const arrayBuffer = await fetchParquetFile(parquetUrl);
    const data = await readParquetColumn(arrayBuffer, filteredColumnNames);
    // Flatten and map to array of objects for chartData
    const seriesNames = [
      "timestamp",
      ...columns.map(({ value }) => value).flat(),
    ];

    const chartData = data.map((row) => {
      const flatRow = row.flat();
      const obj: Record<string, number> = {};
      seriesNames.forEach((key, idx) => {
        obj[key] = flatRow[idx];
      });
      return obj;
    });

    // List of columns that are ignored (e.g., 2D or 3D data)
    const ignoredColumns = Object.entries(info.features)
      .filter(
        ([key, value]) =>
          ["float32", "int32"].includes(value.dtype) && value.shape.length > 1,
      )
      .map(([key]) => key);

    // --- Group columns by scale ---
    // 1. Compute min/max for each column (excluding 'timestamp')
    const numericKeys = seriesNames.filter((k) => k !== "timestamp");
    const colStats: Record<string, { min: number; max: number }> = {};
    numericKeys.forEach((key) => {
      let min = Infinity,
        max = -Infinity;
      for (const row of chartData) {
        const v = row[key];
        if (typeof v === "number" && !isNaN(v)) {
          if (v < min) min = v;
          if (v > max) max = v;
        }
      }
      colStats[key] = { min, max };
    });

    // 2. Group columns by similar scale (log10 range, threshold = 1 order of magnitude)
    const scaleGroups: Record<string, string[]> = {};
    const used = new Set<string>();
    const SCALE_THRESHOLD = 2; // log10(max) - log10(min) within 1 order of magnitude
    for (const key of numericKeys) {
      if (used.has(key)) continue;
      const { min, max } = colStats[key];
      if (!isFinite(min) || !isFinite(max)) continue;
      const logMin = Math.log10(Math.abs(min) + 1e-9);
      const logMax = Math.log10(Math.abs(max) + 1e-9);
      const group: string[] = [key];
      used.add(key);
      for (const other of numericKeys) {
        if (used.has(other) || other === key) continue;
        const { min: omin, max: omax } = colStats[other];
        if (!isFinite(omin) || !isFinite(omax) || omin === omax) continue;
        const ologMin = Math.log10(Math.abs(omin) + 1e-9);
        const ologMax = Math.log10(Math.abs(omax) + 1e-9);
        // If both min/max are within threshold, group together
        if (
          Math.abs(logMin - ologMin) <= SCALE_THRESHOLD &&
          Math.abs(logMax - ologMax) <= SCALE_THRESHOLD
        ) {
          group.push(other);
          used.add(other);
        }
      }
      scaleGroups[key] = group;
    }

    for (const [key, group] of Object.entries(scaleGroups)) {
      scaleGroups[key] = groupIdenticalSeriesNames(group);
    }

    // If any group in chartGroups is longer than 6, split into subgroups of max length 6
    const chartGroups = Object.values(scaleGroups)
      .sort((a, b) => b.length - a.length)
      .flatMap((group) => {
        if (group.length > 6) {
          const subgroups = [];
          for (let i = 0; i < group.length; i += 6) {
            subgroups.push(group.slice(i, i + 6));
          }
          return subgroups;
        }
        return [group];
      });

    const duration = chartData[chartData.length - 1].timestamp;

    const chartDataGroups = chartGroups.map((group) =>
      chartData.map((row) => pick(row, [...group, "timestamp"])),
    );

    return {
      datasetInfo,
      episodeId: episodeId + 1,
      videosInfo,
      chartDataGroups,
      episodes,
      ignoredColumns,
      duration,
    };
  } catch (err) {
    console.error("Error loading episode data:", err);
    throw err;
  }
}

// Safe wrapper for UI error display
export async function getEpisodeDataSafe(
  org: string,
  dataset: string,
  episodeId: number,
): Promise<{ data?: any; error?: string }> {
  try {
    const data = await getEpisodeData(org, dataset, episodeId);
    return { data };
  } catch (err: any) {
    // Only expose the error message, not stack or sensitive info
    return { error: err?.message || String(err) || "Unknown error" };
  }
}

function groupIdenticalSeriesNames(seriesNames: string[]): string[] {
  const seenSuffixes = new Set<string>();
  const suffixMap = new Map<string, string[]>();

  // Build a map from suffix to all items with that suffix (preserve order)
  for (const name of seriesNames) {
    const parts = name.split(SERIES_NAME_DELIMITER);
    const suffix = parts[1] || "";
    if (!suffixMap.has(suffix)) {
      suffixMap.set(suffix, []);
    }
    suffixMap.get(suffix)!.push(name);
  }

  const result: string[] = [];
  for (const name of seriesNames) {
    const parts = name.split(SERIES_NAME_DELIMITER);
    const suffix = parts[1] || "";
    if (!seenSuffixes.has(suffix)) {
      // Insert all items with this suffix
      result.push(...suffixMap.get(suffix)!);
      seenSuffixes.add(suffix);
    }
    // else: already inserted as part of a group, skip
  }
  return result;
}
