"use client";

import { useEffect, useState } from "react";
import { useTime } from "../context/time-context";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

type DataGraphProps = {
  data: Array<Array<Record<string, number>>>;
  onChartsReady?: () => void;
};

import React, { useMemo } from "react";

export const DataRecharts = React.memo(
  ({ data, onChartsReady }: DataGraphProps) => {
    // Shared hoveredTime for all graphs
    const [hoveredTime, setHoveredTime] = useState<number | null>(null);

    if (!Array.isArray(data) || data.length === 0) return null;

    useEffect(() => {
      if (typeof onChartsReady === "function") {
        onChartsReady();
      }
    }, [onChartsReady]);

    return (
      <div className="grid md:grid-cols-2 grid-cols-1 gap-4">
        {data.map((group, idx) => (
          <SingleDataGraph
            key={idx}
            data={group}
            hoveredTime={hoveredTime}
            setHoveredTime={setHoveredTime}
          />
        ))}
      </div>
    );
  },
);

const NESTED_KEY_DELIMITER = ",";

const SingleDataGraph = React.memo(
  ({
    data,
    hoveredTime,
    setHoveredTime,
  }: {
    data: Array<Record<string, number>>;
    hoveredTime: number | null;
    setHoveredTime: (t: number | null) => void;
  }) => {
    const { currentTime, setCurrentTime } = useTime();
    function flattenRow(row: Record<string, any>, prefix = ""): Record<string, number> {
      const result: Record<string, number> = {};
      for (const [key, value] of Object.entries(row)) {
        // Special case: if this is a group value that is a primitive, assign to prefix.key
        if (typeof value === "number") {
          if (prefix) {
            result[`${prefix}${NESTED_KEY_DELIMITER}${key}`] = value;
          } else {
            result[key] = value;
          }
        } else if (value !== null && typeof value === "object" && !Array.isArray(value)) {
          // If it's an object, recurse
          Object.assign(result, flattenRow(value, prefix ? `${prefix}${NESTED_KEY_DELIMITER}${key}` : key));
        }
      }
      // Always keep timestamp at top level if present
      if ("timestamp" in row) {
        result["timestamp"] = row["timestamp"];
      }
      return result;
    }

    // Flatten all rows for recharts
    const chartData = useMemo(() => data.map(row => flattenRow(row)), [data]);
    const [dataKeys, setDataKeys] = useState<string[]>([]);
    const [visibleKeys, setVisibleKeys] = useState<string[]>([]);

    useEffect(() => {
      if (!chartData || chartData.length === 0) return;
      // Get all keys except timestamp from the first row
      const keys = Object.keys(chartData[0]).filter((k) => k !== "timestamp");
      setDataKeys(keys);
      setVisibleKeys(keys);
    }, [chartData]);

    // Parse dataKeys into groups (dot notation)
    const groups: Record<string, string[]> = {};
    const singles: string[] = [];
    dataKeys.forEach((key) => {
      const parts = key.split(NESTED_KEY_DELIMITER);
      if (parts.length > 1) {
        const group = parts[0];
        if (!groups[group]) groups[group] = [];
        groups[group].push(key);
      } else {
        singles.push(key);
      }
    });

    // Assign a color per group (and for singles)
    const allGroups = [...Object.keys(groups), ...singles];
    const groupColorMap: Record<string, string> = {};
    allGroups.forEach((group, idx) => {
      groupColorMap[group] = `hsl(${idx * (360 / allGroups.length)}, 100%, 50%)`;
    });

    // Find the closest data point to the current time for highlighting
    const findClosestDataIndex = (time: number) => {
      if (!chartData.length) return 0;
      // Find the index of the first data point whose timestamp is >= time (ceiling)
      const idx = chartData.findIndex((point) => point.timestamp >= time);
      if (idx !== -1) return idx;
      // If all timestamps are less than time, return the last index
      return chartData.length - 1;
    };

    const handleMouseLeave = () => {
      setHoveredTime(null);
    };

    const handleClick = (data: any) => {
      if (data && data.activePayload && data.activePayload.length) {
        const timeValue = data.activePayload[0].payload.timestamp;
        setCurrentTime(timeValue);
      }
    };

    // Custom legend to show current value next to each series
    const CustomLegend = () => {
      const closestIndex = findClosestDataIndex(
        hoveredTime != null ? hoveredTime : currentTime,
      );
      const currentData = chartData[closestIndex] || {};

      // Parse dataKeys into groups (dot notation)
      const groups: Record<string, string[]> = {};
      const singles: string[] = [];
      dataKeys.forEach((key) => {
        const parts = key.split(NESTED_KEY_DELIMITER);
        if (parts.length > 1) {
          const group = parts[0];
          if (!groups[group]) groups[group] = [];
          groups[group].push(key);
        } else {
          singles.push(key);
        }
      });

      // Assign a color per group (and for singles)
      const allGroups = [...Object.keys(groups), ...singles];
      const groupColorMap: Record<string, string> = {};
      allGroups.forEach((group, idx) => {
        groupColorMap[group] = `hsl(${idx * (360 / allGroups.length)}, 100%, 50%)`;
      });

      const isGroupChecked = (group: string) => groups[group].every(k => visibleKeys.includes(k));
      const isGroupIndeterminate = (group: string) => groups[group].some(k => visibleKeys.includes(k)) && !isGroupChecked(group);

      const handleGroupCheckboxChange = (group: string) => {
        if (isGroupChecked(group)) {
          // Uncheck all children
          setVisibleKeys((prev) => prev.filter(k => !groups[group].includes(k)));
        } else {
          // Check all children
          setVisibleKeys((prev) => Array.from(new Set([...prev, ...groups[group]])));
        }
      };

      const handleCheckboxChange = (key: string) => {
        setVisibleKeys((prev) =>
          prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]
        );
      };

      return (
        <div className="grid grid-cols-[repeat(auto-fit,250px)] gap-4 mx-8">
          {/* Grouped keys */}
          {Object.entries(groups).map(([group, children]) => {
            const color = groupColorMap[group];
            return (
              <div key={group} className="mb-2">
                <label className="flex gap-2 cursor-pointer select-none font-semibold">
                  <input
                    type="checkbox"
                    checked={isGroupChecked(group)}
                    ref={el => { if (el) el.indeterminate = isGroupIndeterminate(group); }}
                    onChange={() => handleGroupCheckboxChange(group)}
                    className="size-3.5 mt-1"
                    style={{ accentColor: color }}
                  />
                  <span className="text-sm w-40 text-white">{group}</span>
                </label>
                <div className="pl-7 flex flex-col gap-1 mt-1">
                  {children.map((key) => (
                    <label key={key} className="flex gap-2 cursor-pointer select-none">
                      <input
                        type="checkbox"
                        checked={visibleKeys.includes(key)}
                        onChange={() => handleCheckboxChange(key)}
                        className="size-3.5 mt-1"
                        style={{ accentColor: color }}
                      />
                      <span className={`text-xs break-all w-36 ${visibleKeys.includes(key) ? "text-white" : "text-gray-400"}`}>{key.slice(group.length + 1)}</span>
                      <span className={`text-xs font-mono ml-auto ${visibleKeys.includes(key) ? "text-orange-300" : "text-gray-500"}`}>
                        {typeof currentData[key] === "number" ? currentData[key].toFixed(2) : "--"}
                      </span>
                    </label>
                  ))}
                </div>
              </div>
            );
          })}
          {/* Singles (non-grouped) */}
          {singles.map((key) => {
            const color = groupColorMap[key];
            return (
              <label key={key} className="flex gap-2 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={visibleKeys.includes(key)}
                  onChange={() => handleCheckboxChange(key)}
                  className="size-3.5 mt-1"
                  style={{ accentColor: color }}
                />
                <span className={`text-sm break-all w-40 ${visibleKeys.includes(key) ? "text-white" : "text-gray-400"}`}>{key}</span>
                <span className={`text-sm font-mono ml-auto ${visibleKeys.includes(key) ? "text-orange-300" : "text-gray-500"}`}>
                  {typeof currentData[key] === "number" ? currentData[key].toFixed(2) : "--"}
                </span>
              </label>
            );
          })}
        </div>
      );
    };

    return (
      <div className="w-full">
        <div className="w-full h-80" onMouseLeave={handleMouseLeave}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={chartData}
              syncId="episode-sync"
              margin={{ top: 24, right: 16, left: 0, bottom: 16 }}
              onClick={handleClick}
              onMouseMove={(state: any) => {
                setHoveredTime(
                  state?.activePayload?.[0]?.payload?.timestamp ??
                    state?.activeLabel ??
                    null,
                );
              }}
              onMouseLeave={handleMouseLeave}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#444" />
              <XAxis
                dataKey="timestamp"
                label={{
                  value: "time",
                  position: "insideBottomLeft",
                  fill: "#cbd5e1",
                }}
                domain={[
                  chartData.at(0)?.timestamp ?? 0,
                  chartData.at(-1)?.timestamp ?? 0,
                ]}
                ticks={useMemo(
                  () =>
                    Array.from(
                      new Set(chartData.map((d) => Math.ceil(d.timestamp))),
                    ),
                  [chartData],
                )}
                stroke="#cbd5e1"
                minTickGap={20} // Increased for fewer ticks
                allowDataOverflow={true}
              />
              <YAxis
                domain={["auto", "auto"]}
                stroke="#cbd5e1"
                interval={0}
                allowDataOverflow={true}
              />

              <Tooltip
                content={() => null}
                active={true}
                isAnimationActive={false}
                defaultIndex={
                  !hoveredTime ? findClosestDataIndex(currentTime) : undefined
                }
              />

              {/* Render lines for visible dataKeys only */}
              {dataKeys.map((key) => {
                // Use group color for all keys in a group
                const group = key.includes(NESTED_KEY_DELIMITER) ? key.split(NESTED_KEY_DELIMITER)[0] : key;
                const color = groupColorMap[group];
                let strokeDasharray: string | undefined = undefined;
                if (groups[group] && groups[group].length > 1) {
                  const idxInGroup = groups[group].indexOf(key);
                  if (idxInGroup > 0) strokeDasharray = "5 5";
                }
                return (
                  visibleKeys.includes(key) && (
                    <Line
                      key={key}
                      type="monotone"
                      dataKey={key}
                      name={key}
                      stroke={color}
                      strokeDasharray={strokeDasharray}
                      dot={false}
                      activeDot={false}
                      strokeWidth={1.5}
                      isAnimationActive={false}
                    />
                  )
                );
              })}
            </LineChart>
          </ResponsiveContainer>
        </div>
        <CustomLegend />
      </div>
    );
  },
); // End React.memo

SingleDataGraph.displayName = "SingleDataGraph";
DataRecharts.displayName = "DataGraph";
export default DataRecharts;
