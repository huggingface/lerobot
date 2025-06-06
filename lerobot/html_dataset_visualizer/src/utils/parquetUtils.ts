import { parquetRead } from "hyparquet";

export interface DatasetMetadata {
  codebase_version: string;
  robot_type: string;
  total_episodes: number;
  total_frames: number;
  total_tasks: number;
  total_videos: number;
  total_chunks: number;
  chunks_size: number;
  fps: number;
  splits: Record<string, string>;
  data_path: string;
  video_path: string;
  features: Record<
    string,
    {
      dtype: string;
      shape: any[];
      names: any[] | Record<string, any> | null;
      info?: Record<string, any>;
    }
  >;
}

export async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(
      `Failed to fetch JSON ${url}: ${res.status} ${res.statusText}`,
    );
  }
  return res.json() as Promise<T>;
}

export function formatStringWithVars(
  format: string,
  vars: Record<string, any>,
): string {
  return format.replace(/{(\w+)(?::\d+d)?}/g, (_, key) => vars[key]);
}

// Fetch and parse the Parquet file
export async function fetchParquetFile(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url);
  return res.arrayBuffer();
}

// Read specific columns from the Parquet file
export async function readParquetColumn(
  fileBuffer: ArrayBuffer,
  columns: string[],
): Promise<any[]> {
  return new Promise((resolve) => {
    parquetRead({
      file: fileBuffer,
      columns,
      onComplete: (data: any[]) => resolve(data),
    });
  });
}

// Convert a 2D array to a CSV string
export function arrayToCSV(data: (number | string)[][]): string {
  return data.map((row) => row.join(",")).join("\n");
}

// Get rows from the current frame data
export function getRows(currentFrameData: any[], columns: any[]) {
  if (!currentFrameData || currentFrameData.length === 0) {
    return [];
  }

  const rows = [];
  const nRows = Math.max(...columns.map((column) => column.value.length));
  let rowIndex = 0;

  while (rowIndex < nRows) {
    const row = [];
    // number of states may NOT match number of actions. In this case, we null-pad the 2D array
    const nullCell = { isNull: true };
    // row consists of [state value, action value]
    let idx = rowIndex;

    for (const column of columns) {
      const nColumn = column.value.length;
      row.push(rowIndex < nColumn ? currentFrameData[idx] : nullCell);
      idx += nColumn; // because currentFrameData = [state0, state1, ..., stateN, action0, action1, ..., actionN]
    }

    rowIndex += 1;
    rows.push(row);
  }

  return rows;
}
