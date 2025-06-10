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
    const episodes =
      process.env.EPISODES === undefined
        ? Array.from(
            { length: datasetInfo.total_episodes },
            // episode id starts from 0
            (_, i) => i,
          )
        : process.env.EPISODES
            .split(/\s+/)
            .map((x) => parseInt(x.trim(), 10))
            .filter((x) => !isNaN(x));

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

    // 1. Group all numeric keys by suffix (excluding 'timestamp')
    const numericKeys = seriesNames.filter((k) => k !== "timestamp");
    const suffixGroupsMap: Record<string, string[]> = {};
    for (const key of numericKeys) {
      const parts = key.split(SERIES_NAME_DELIMITER);
      const suffix = parts[1] || parts[0]; // fallback to key if no delimiter
      if (!suffixGroupsMap[suffix]) suffixGroupsMap[suffix] = [];
      suffixGroupsMap[suffix].push(key);
    }
    const suffixGroups = Object.values(suffixGroupsMap);

    // 2. Compute min/max for each suffix group as a whole
    const groupStats: Record<string, { min: number; max: number }> = {};
    suffixGroups.forEach((group) => {
      let min = Infinity,
        max = -Infinity;
      for (const row of chartData) {
        for (const key of group) {
          const v = row[key];
          if (typeof v === "number" && !isNaN(v)) {
            if (v < min) min = v;
            if (v > max) max = v;
          }
        }
      }
      // Use the first key in the group as the group id
      groupStats[group[0]] = { min, max };
    });

    // 3. Group suffix groups by similar scale (treat each suffix group as a unit)
    const scaleGroups: Record<string, string[][]> = {};
    const used = new Set<string>();
    const SCALE_THRESHOLD = 2;
    for (const group of suffixGroups) {
      const groupId = group[0];
      if (used.has(groupId)) continue;
      const { min, max } = groupStats[groupId];
      if (!isFinite(min) || !isFinite(max)) continue;
      const logMin = Math.log10(Math.abs(min) + 1e-9);
      const logMax = Math.log10(Math.abs(max) + 1e-9);
      const unit: string[][] = [group];
      used.add(groupId);
      for (const other of suffixGroups) {
        const otherId = other[0];
        if (used.has(otherId) || otherId === groupId) continue;
        const { min: omin, max: omax } = groupStats[otherId];
        if (!isFinite(omin) || !isFinite(omax) || omin === omax) continue;
        const ologMin = Math.log10(Math.abs(omin) + 1e-9);
        const ologMax = Math.log10(Math.abs(omax) + 1e-9);
        if (
          Math.abs(logMin - ologMin) <= SCALE_THRESHOLD &&
          Math.abs(logMax - ologMax) <= SCALE_THRESHOLD
        ) {
          unit.push(other);
          used.add(otherId);
        }
      }
      scaleGroups[groupId] = unit;
    }

    // 4. Flatten scaleGroups into chartGroups (array of arrays of keys)
    const chartGroups: string[][] = Object.values(scaleGroups)
      .sort((a, b) => b.length - a.length)
      .flatMap((suffixGroupArr) => {
        // suffixGroupArr is array of suffix groups (each is array of keys)
        const merged = suffixGroupArr.flat();
        if (merged.length > 6) {
          const subgroups = [];
          for (let i = 0; i < merged.length; i += 6) {
            subgroups.push(merged.slice(i, i + 6));
          }
          return subgroups;
        }
        return [merged];
      });

    const duration = chartData[chartData.length - 1].timestamp;

    // Utility: group row keys by suffix
    function groupRowBySuffix(row: Record<string, number>): Record<string, any> {
      const result: Record<string, any> = {};
      const suffixGroups: Record<string, Record<string, number>> = {};
      for (const [key, value] of Object.entries(row)) {
        if (key === "timestamp") {
          result["timestamp"] = value;
          continue;
        }
        const parts = key.split(SERIES_NAME_DELIMITER);
        if (parts.length === 2) {
          const [prefix, suffix] = parts;
          if (!suffixGroups[suffix]) suffixGroups[suffix] = {};
          suffixGroups[suffix][prefix] = value;
        } else {
          result[key] = value;
        }
      }
      for (const [suffix, group] of Object.entries(suffixGroups)) {
        const keys = Object.keys(group);
        if (keys.length === 1) {
          // Use the full original name as the key
          const fullName = `${keys[0]}${SERIES_NAME_DELIMITER}${suffix}`;
          result[fullName] = group[keys[0]];
        } else {
          result[suffix] = group;
        }
      }
      return result;
    }

    const chartDataGroups = chartGroups.map((group) =>
      chartData.map((row) => groupRowBySuffix(pick(row, [...group, "timestamp"])))
    );

    return {
      datasetInfo,
      episodeId,
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
