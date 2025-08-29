import argparse
import contextlib
import json
import os
import shutil
import traceback

import numpy as np
import pandas as pd


def load_jsonl(file_path):
    """
    从JSONL文件加载数据
    (Load data from a JSONL file)

    Args:
        file_path (str): JSONL文件路径 (Path to the JSONL file)

    Returns:
        list: 包含文件中每行JSON对象的列表 (List containing JSON objects from each line)
    """
    data = []

    # Special handling for episodes_stats.jsonl
    if "episodes_stats.jsonl" in file_path:
        try:
            # Try to load the entire file as a JSON array
            with open(file_path) as f:
                content = f.read()
                # Check if the content starts with '[' and ends with ']'
                if content.strip().startswith("[") and content.strip().endswith("]"):
                    return json.loads(content)
                else:
                    # Try to add brackets and parse
                    try:
                        return json.loads("[" + content + "]")
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"Error loading {file_path} as JSON array: {e}")

        # Fall back to line-by-line parsing
        try:
            with open(file_path) as f:
                for line in f:
                    if line.strip():
                        with contextlib.suppress(json.JSONDecodeError):
                            data.append(json.loads(line))
        except Exception as e:
            print(f"Error loading {file_path} line by line: {e}")
    else:
        # Standard JSONL parsing for other files
        with open(file_path) as f:
            for line in f:
                if line.strip():
                    with contextlib.suppress(json.JSONDecodeError):
                        data.append(json.loads(line))

    return data


def save_jsonl(data, file_path):
    """
    将数据保存为JSONL格式
    (Save data in JSONL format)

    Args:
        data (list): 要保存的JSON对象列表 (List of JSON objects to save)
        file_path (str): 输出文件路径 (Path to the output file)
    """
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def merge_stats(stats_list):
    """
    合并多个数据集的统计信息，确保维度一致性
    (Merge statistics from multiple datasets, ensuring dimensional consistency)

    Args:
        stats_list (list): 包含每个数据集统计信息的字典列表
                          (List of dictionaries containing statistics for each dataset)

    Returns:
        dict: 合并后的统计信息 (Merged statistics)
    """
    # Initialize merged stats with the structure of the first stats
    merged_stats = {}

    # Find common features across all stats
    common_features = set(stats_list[0].keys())
    for stats in stats_list[1:]:
        common_features = common_features.intersection(set(stats.keys()))

    # Process features in the order they appear in the first stats file
    for feature in stats_list[0]:
        if feature not in common_features:
            continue

        merged_stats[feature] = {}

        # Find common stat types for this feature
        common_stat_types = []
        for stat_type in ["mean", "std", "max", "min"]:
            if all(stat_type in stats[feature] for stats in stats_list):
                common_stat_types.append(stat_type)

        # Determine the original shape of each value
        original_shapes = []
        for stats in stats_list:
            if "mean" in stats[feature]:
                shape = np.array(stats[feature]["mean"]).shape
                original_shapes.append(shape)

        # Special handling for image features to preserve nested structure
        if feature.startswith("observation.images."):
            for stat_type in common_stat_types:
                try:
                    # Get all values
                    values = [stats[feature][stat_type] for stats in stats_list]

                    # For image features, we need to preserve the nested structure
                    # Initialize with the first value's structure
                    result = []

                    # For RGB channels
                    for channel_idx in range(len(values[0])):
                        channel_result = []

                        # For each pixel row
                        for pixel_idx in range(len(values[0][channel_idx])):
                            pixel_result = []

                            # For each pixel value
                            for value_idx in range(len(values[0][channel_idx][pixel_idx])):
                                # Calculate statistic based on type
                                if stat_type == "mean":
                                    # Simple average
                                    avg = sum(
                                        values[i][channel_idx][pixel_idx][value_idx]
                                        for i in range(len(values))
                                    ) / len(values)
                                    pixel_result.append(avg)
                                elif stat_type == "std":
                                    # Simple average of std
                                    avg = sum(
                                        values[i][channel_idx][pixel_idx][value_idx]
                                        for i in range(len(values))
                                    ) / len(values)
                                    pixel_result.append(avg)
                                elif stat_type == "max":
                                    # Maximum
                                    max_val = max(
                                        values[i][channel_idx][pixel_idx][value_idx]
                                        for i in range(len(values))
                                    )
                                    pixel_result.append(max_val)
                                elif stat_type == "min":
                                    # Minimum
                                    min_val = min(
                                        values[i][channel_idx][pixel_idx][value_idx]
                                        for i in range(len(values))
                                    )
                                    pixel_result.append(min_val)

                            channel_result.append(pixel_result)

                        result.append(channel_result)

                    merged_stats[feature][stat_type] = result
                except Exception as e:
                    print(f"Warning: Error processing image feature {feature}.{stat_type}: {e}")
                    # Fallback to first value
                    merged_stats[feature][stat_type] = values[0]
        # If all shapes are the same, no need for special handling
        elif len({str(shape) for shape in original_shapes}) == 1:
            # All shapes are the same, use standard merging
            for stat_type in common_stat_types:
                values = [stats[feature][stat_type] for stats in stats_list]

                try:
                    # Calculate the new statistic based on the type
                    if stat_type == "mean":
                        if all("count" in stats[feature] for stats in stats_list):
                            counts = [stats[feature]["count"][0] for stats in stats_list]
                            total_count = sum(counts)
                            weighted_values = [
                                np.array(val) * count / total_count
                                for val, count in zip(values, counts, strict=False)
                            ]
                            merged_stats[feature][stat_type] = np.sum(weighted_values, axis=0).tolist()
                        else:
                            merged_stats[feature][stat_type] = np.mean(np.array(values), axis=0).tolist()

                    elif stat_type == "std":
                        if all("count" in stats[feature] for stats in stats_list):
                            counts = [stats[feature]["count"][0] for stats in stats_list]
                            total_count = sum(counts)
                            variances = [np.array(std) ** 2 for std in values]
                            weighted_variances = [
                                var * count / total_count
                                for var, count in zip(variances, counts, strict=False)
                            ]
                            merged_stats[feature][stat_type] = np.sqrt(
                                np.sum(weighted_variances, axis=0)
                            ).tolist()
                        else:
                            merged_stats[feature][stat_type] = np.mean(np.array(values), axis=0).tolist()

                    elif stat_type == "max":
                        merged_stats[feature][stat_type] = np.maximum.reduce(np.array(values)).tolist()

                    elif stat_type == "min":
                        merged_stats[feature][stat_type] = np.minimum.reduce(np.array(values)).tolist()
                except Exception as e:
                    print(f"Warning: Error processing {feature}.{stat_type}: {e}")
                    continue
        else:
            # Shapes are different, need special handling for state vectors
            if feature in ["observation.state", "action"]:
                # For state vectors, we need to handle different dimensions
                max_dim = max(len(np.array(stats[feature]["mean"]).flatten()) for stats in stats_list)

                for stat_type in common_stat_types:
                    try:
                        # Get values and their original dimensions
                        values_with_dims = []
                        for stats in stats_list:
                            val = np.array(stats[feature][stat_type]).flatten()
                            dim = len(val)
                            values_with_dims.append((val, dim))

                        # Initialize result array with zeros
                        result = np.zeros(max_dim)

                        # Calculate statistics for each dimension separately
                        if stat_type == "mean":
                            if all("count" in stats[feature] for stats in stats_list):
                                counts = [stats[feature]["count"][0] for stats in stats_list]
                                total_count = sum(counts)

                                # For each dimension, calculate weighted mean of available values
                                for d in range(max_dim):
                                    dim_values = []
                                    dim_weights = []
                                    for (val, dim), count in zip(values_with_dims, counts, strict=False):
                                        if d < dim:  # Only use values that have this dimension
                                            dim_values.append(val[d])
                                            dim_weights.append(count)

                                    if dim_values:  # If we have values for this dimension
                                        weighted_sum = sum(
                                            v * w for v, w in zip(dim_values, dim_weights, strict=False)
                                        )
                                        result[d] = weighted_sum / sum(dim_weights)
                            else:
                                # Simple average for each dimension
                                for d in range(max_dim):
                                    dim_values = [val[d] for val, dim in values_with_dims if d < dim]
                                    if dim_values:
                                        result[d] = sum(dim_values) / len(dim_values)

                        elif stat_type == "std":
                            if all("count" in stats[feature] for stats in stats_list):
                                counts = [stats[feature]["count"][0] for stats in stats_list]
                                total_count = sum(counts)

                                # For each dimension, calculate weighted variance
                                for d in range(max_dim):
                                    dim_variances = []
                                    dim_weights = []
                                    for (val, dim), count in zip(values_with_dims, counts, strict=False):
                                        if d < dim:  # Only use values that have this dimension
                                            dim_variances.append(val[d] ** 2)  # Square for variance
                                            dim_weights.append(count)

                                    if dim_variances:  # If we have values for this dimension
                                        weighted_var = sum(
                                            v * w for v, w in zip(dim_variances, dim_weights, strict=False)
                                        ) / sum(dim_weights)
                                        result[d] = np.sqrt(weighted_var)  # Take sqrt for std
                            else:
                                # Simple average of std for each dimension
                                for d in range(max_dim):
                                    dim_values = [val[d] for val, dim in values_with_dims if d < dim]
                                    if dim_values:
                                        result[d] = sum(dim_values) / len(dim_values)

                        elif stat_type == "max":
                            # For each dimension, take the maximum of available values
                            for d in range(max_dim):
                                dim_values = [val[d] for val, dim in values_with_dims if d < dim]
                                if dim_values:
                                    result[d] = max(dim_values)

                        elif stat_type == "min":
                            # For each dimension, take the minimum of available values
                            for d in range(max_dim):
                                dim_values = [val[d] for val, dim in values_with_dims if d < dim]
                                if dim_values:
                                    result[d] = min(dim_values)

                        # Convert result to list and store
                        merged_stats[feature][stat_type] = result.tolist()

                    except Exception as e:
                        print(
                            f"Warning: Error processing {feature}.{stat_type} with different dimensions: {e}"
                        )
                        continue
            else:
                # For other features with different shapes, use the first shape as template
                template_shape = original_shapes[0]
                print(f"Using shape {template_shape} as template for {feature}")

                for stat_type in common_stat_types:
                    try:
                        # Use the first stats as template
                        merged_stats[feature][stat_type] = stats_list[0][feature][stat_type]
                    except Exception as e:
                        print(
                            f"Warning: Error processing {feature}.{stat_type} with shape {template_shape}: {e}"
                        )
                        continue

        # Add count if available in all stats
        if all("count" in stats[feature] for stats in stats_list):
            try:
                merged_stats[feature]["count"] = [sum(stats[feature]["count"][0] for stats in stats_list)]
            except Exception as e:
                print(f"Warning: Error processing {feature}.count: {e}")

    return merged_stats


def copy_videos(source_folders, output_folder, episode_mapping):
    """
    从源文件夹复制视频文件到输出文件夹，保持正确的索引和结构
    (Copy video files from source folders to output folder, maintaining correct indices and structure)

    Args:
        source_folders (list): 源数据集文件夹路径列表 (List of source dataset folder paths)
        output_folder (str): 输出文件夹路径 (Output folder path)
        episode_mapping (list): 包含(旧文件夹,旧索引,新索引)元组的列表
                               (List of tuples containing (old_folder, old_index, new_index))
    """
    # Get info.json to determine video structure
    info_path = os.path.join(source_folders[0], "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)

    video_path_template = info["video_path"]

    # Identify video keys from the template
    # Example: "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    video_keys = []
    for feature_name, feature_info in info["features"].items():
        if feature_info.get("dtype") == "video":
            # Use the full feature name as the video key
            video_keys.append(feature_name)

    print(f"Found video keys: {video_keys}")

    # Copy videos for each episode
    for old_folder, old_index, new_index in episode_mapping:
        # Determine episode chunk (usually 0 for small datasets)
        episode_chunk = old_index // info["chunks_size"]
        new_episode_chunk = new_index // info["chunks_size"]

        for video_key in video_keys:
            # Try different possible source paths
            source_patterns = [
                # Standard path with the episode index from metadata
                os.path.join(
                    old_folder,
                    video_path_template.format(
                        episode_chunk=episode_chunk, video_key=video_key, episode_index=old_index
                    ),
                ),
                # Try with 0-based indexing
                os.path.join(
                    old_folder,
                    video_path_template.format(episode_chunk=0, video_key=video_key, episode_index=0),
                ),
                # Try with different formatting
                os.path.join(
                    old_folder, f"videos/chunk-{episode_chunk:03d}/{video_key}/episode_{old_index}.mp4"
                ),
                os.path.join(old_folder, f"videos/chunk-000/{video_key}/episode_000000.mp4"),
            ]

            # Find the first existing source path
            source_video_path = None
            for pattern in source_patterns:
                if os.path.exists(pattern):
                    source_video_path = pattern
                    break

            if source_video_path:
                # Construct destination path
                dest_video_path = os.path.join(
                    output_folder,
                    video_path_template.format(
                        episode_chunk=new_episode_chunk, video_key=video_key, episode_index=new_index
                    ),
                )

                # Create destination directory if it doesn't exist
                os.makedirs(os.path.dirname(dest_video_path), exist_ok=True)

                print(f"Copying video: {source_video_path} -> {dest_video_path}")
                shutil.copy2(source_video_path, dest_video_path)
            else:
                # If no file is found, search the directory recursively
                found = False
                for root, _, files in os.walk(os.path.join(old_folder, "videos")):
                    for file in files:
                        if file.endswith(".mp4") and video_key in root:
                            source_video_path = os.path.join(root, file)

                            # Construct destination path
                            dest_video_path = os.path.join(
                                output_folder,
                                video_path_template.format(
                                    episode_chunk=new_episode_chunk,
                                    video_key=video_key,
                                    episode_index=new_index,
                                ),
                            )

                            # Create destination directory if it doesn't exist
                            os.makedirs(os.path.dirname(dest_video_path), exist_ok=True)

                            print(
                                f"Copying video (found by search): {source_video_path} -> {dest_video_path}"
                            )
                            shutil.copy2(source_video_path, dest_video_path)
                            found = True
                            break
                    if found:
                        break

                if not found:
                    print(
                        f"Warning: Video file not found for {video_key}, episode {old_index} in {old_folder}"
                    )


def validate_timestamps(source_folders, tolerance_s=1e-4):
    """
    验证源数据集的时间戳结构，识别潜在问题
    (Validate timestamp structure of source datasets, identify potential issues)

    Args:
        source_folders (list): 源数据集文件夹路径列表 (List of source dataset folder paths)
        tolerance_s (float): 时间戳不连续性的容差值，以秒为单位 (Tolerance for timestamp discontinuities in seconds)

    Returns:
        tuple: (issues, fps_values) - 问题列表和检测到的FPS值列表
               (List of issues and list of detected FPS values)
    """
    issues = []
    fps_values = []

    for folder in source_folders:
        try:
            # 尝试从 info.json 获取 FPS (Try to get FPS from info.json)
            info_path = os.path.join(folder, "meta", "info.json")
            if os.path.exists(info_path):
                with open(info_path) as f:
                    info = json.load(f)
                    if "fps" in info:
                        fps = info["fps"]
                        fps_values.append(fps)
                        print(f"数据集 {folder} FPS={fps} (Dataset {folder} FPS={fps})")

            # 检查是否有parquet文件包含时间戳 (Check if any parquet files contain timestamps)
            parquet_path = None
            for root, _, files in os.walk(os.path.join(folder, "parquet")):
                for file in files:
                    if file.endswith(".parquet"):
                        parquet_path = os.path.join(root, file)
                        break
                if parquet_path:
                    break

            if not parquet_path:
                for root, _, files in os.walk(os.path.join(folder, "data")):
                    for file in files:
                        if file.endswith(".parquet"):
                            parquet_path = os.path.join(root, file)
                            break
                    if parquet_path:
                        break

            if parquet_path:
                df = pd.read_parquet(parquet_path)
                timestamp_cols = [col for col in df.columns if "timestamp" in col or "time" in col]
                if timestamp_cols:
                    print(
                        f"数据集 {folder} 包含时间戳列: {timestamp_cols} (Dataset {folder} contains timestamp columns: {timestamp_cols})"
                    )
                else:
                    issues.append(
                        f"警告: 数据集 {folder} 没有时间戳列 (Warning: Dataset {folder} has no timestamp columns)"
                    )
            else:
                issues.append(
                    f"警告: 数据集 {folder} 未找到parquet文件 (Warning: No parquet files found in dataset {folder})"
                )

        except Exception as e:
            issues.append(
                f"错误: 验证数据集 {folder} 失败: {e} (Error: Failed to validate dataset {folder}: {e})"
            )
            print(f"验证错误: {e} (Validation error: {e})")
            traceback.print_exc()

    # 检查FPS是否一致 (Check if FPS values are consistent)
    if len(set(fps_values)) > 1:
        issues.append(
            f"警告: 数据集FPS不一致: {fps_values} (Warning: Inconsistent FPS across datasets: {fps_values})"
        )

    return issues, fps_values


def copy_data_files(
    source_folders,
    output_folder,
    episode_mapping,
    max_dim=18,
    fps=None,
    episode_to_frame_index=None,
    folder_task_mapping=None,
    chunks_size=1000,
    default_fps=20,
):
    """
    复制并处理parquet数据文件，包括维度填充和索引更新
    (Copy and process parquet data files, including dimension padding and index updates)

    Args:
        source_folders (list): 源数据集文件夹路径列表 (List of source dataset folder paths)
        output_folder (str): 输出文件夹路径 (Output folder path)
        episode_mapping (list): 包含(旧文件夹,旧索引,新索引)元组的列表
                               (List of tuples containing (old_folder, old_index, new_index))
        max_dim (int): 向量的最大维度 (Maximum dimension for vectors)
        fps (float, optional): 帧率，如果未提供则从第一个数据集获取 (Frame rate, if not provided will be obtained from the first dataset)
        episode_to_frame_index (dict, optional): 每个新episode索引对应的起始帧索引映射
                                               (Mapping of each new episode index to its starting frame index)
        folder_task_mapping (dict, optional): 每个文件夹中task_index的映射关系
                                            (Mapping of task_index for each folder)
        chunks_size (int): 每个chunk包含的episode数量 (Number of episodes per chunk)
        default_fps (float): 默认帧率，当无法从数据集获取时使用 (Default frame rate when unable to obtain from dataset)

    Returns:
        bool: 是否成功复制了至少一个文件 (Whether at least one file was successfully copied)
    """
    # 获取第一个数据集的FPS（如果未提供）(Get FPS from first dataset if not provided)
    if fps is None:
        info_path = os.path.join(source_folders[0], "meta", "info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
                fps = info.get(
                    "fps", default_fps
                )  # 使用变量替代硬编码的20 (Use variable instead of hardcoded 20)
        else:
            fps = default_fps  # 使用变量替代硬编码的20 (Use variable instead of hardcoded 20)

    print(f"使用FPS={fps} (Using FPS={fps})")

    # 为每个episode复制和处理数据文件 (Copy and process data files for each episode)
    total_copied = 0
    total_failed = 0

    # 添加一个列表来记录失败的文件及原因
    # (Add a list to record failed files and reasons)
    failed_files = []

    for i, (old_folder, old_index, new_index) in enumerate(episode_mapping):
        # 尝试找到源parquet文件 (Try to find source parquet file)
        episode_str = f"episode_{old_index:06d}.parquet"
        source_paths = [
            os.path.join(old_folder, "parquet", episode_str),
            os.path.join(old_folder, "data", episode_str),
        ]

        source_path = None
        for path in source_paths:
            if os.path.exists(path):
                source_path = path
                break

        if source_path:
            try:
                # 读取parquet文件 (Read parquet file)
                df = pd.read_parquet(source_path)

                # 检查是否需要填充维度 (Check if dimensions need padding)
                for feature in ["observation.state", "action"]:
                    if feature in df.columns:
                        # 检查第一个非空值 (Check first non-null value)
                        for _idx, value in enumerate(df[feature]):
                            if value is not None and isinstance(value, (list, np.ndarray)):
                                current_dim = len(value)
                                if current_dim < max_dim:
                                    print(
                                        f"填充 {feature} 从 {current_dim} 维到 {max_dim} 维 (Padding {feature} from {current_dim} to {max_dim} dimensions)"
                                    )
                                    # 使用零填充到目标维度 (Pad with zeros to target dimension)
                                    df[feature] = df[feature].apply(
                                        lambda x: np.pad(x, (0, max_dim - len(x)), "constant").tolist()
                                        if x is not None
                                        and isinstance(x, (list, np.ndarray))
                                        and len(x) < max_dim
                                        else x
                                    )
                                break

                # 更新episode_index列 (Update episode_index column)
                if "episode_index" in df.columns:
                    print(
                        f"更新episode_index从 {df['episode_index'].iloc[0]} 到 {new_index} (Update episode_index from {df['episode_index'].iloc[0]} to {new_index})"
                    )
                    df["episode_index"] = new_index

                # 更新index列 (Update index column)
                if "index" in df.columns:
                    if episode_to_frame_index and new_index in episode_to_frame_index:
                        # 使用预先计算的帧索引起始值 (Use pre-calculated frame index start value)
                        first_index = episode_to_frame_index[new_index]
                        print(
                            f"更新index列，起始值: {first_index}（使用全局累积帧计数）(Update index column, start value: {first_index} (using global cumulative frame count))"
                        )
                    else:
                        # 如果没有提供映射，使用当前的计算方式作为回退
                        # (If no mapping provided, use current calculation as fallback)
                        first_index = new_index * len(df)
                        print(
                            f"更新index列，起始值: {first_index}（使用episode索引乘以长度）(Update index column, start value: {first_index} (using episode index multiplied by length))"
                        )

                    # 更新所有帧的索引 (Update indices for all frames)
                    df["index"] = [first_index + i for i in range(len(df))]

                # 更新task_index列 (Update task_index column)
                if "task_index" in df.columns and folder_task_mapping and old_folder in folder_task_mapping:
                    # 获取当前task_index (Get current task_index)
                    current_task_index = df["task_index"].iloc[0]

                    # 检查是否有对应的新索引 (Check if there's a corresponding new index)
                    if current_task_index in folder_task_mapping[old_folder]:
                        new_task_index = folder_task_mapping[old_folder][current_task_index]
                        print(
                            f"更新task_index从 {current_task_index} 到 {new_task_index} (Update task_index from {current_task_index} to {new_task_index})"
                        )
                        df["task_index"] = new_task_index
                    else:
                        print(
                            f"警告: 找不到task_index {current_task_index}的映射关系 (Warning: No mapping found for task_index {current_task_index})"
                        )

                # 计算chunk编号 (Calculate chunk number)
                chunk_index = new_index // chunks_size

                # 创建正确的目标目录 (Create correct target directory)
                chunk_dir = os.path.join(output_folder, "data", f"chunk-{chunk_index:03d}")
                os.makedirs(chunk_dir, exist_ok=True)

                # 构建正确的目标路径 (Build correct target path)
                dest_path = os.path.join(chunk_dir, f"episode_{new_index:06d}.parquet")

                # 保存到正确位置 (Save to correct location)
                df.to_parquet(dest_path, index=False)

                total_copied += 1
                print(f"已处理并保存: {dest_path} (Processed and saved: {dest_path})")

            except Exception as e:
                error_msg = f"处理 {source_path} 失败: {e} (Processing {source_path} failed: {e})"
                print(error_msg)
                traceback.print_exc()
                failed_files.append({"file": source_path, "reason": str(e), "episode": old_index})
                total_failed += 1
        else:
            # 文件不在标准位置，尝试递归搜索
            found = False
            for root, _, files in os.walk(old_folder):
                for file in files:
                    if file.endswith(".parquet") and f"episode_{old_index:06d}" in file:
                        try:
                            source_path = os.path.join(root, file)

                            # 读取parquet文件 (Read parquet file)
                            df = pd.read_parquet(source_path)

                            # 检查是否需要填充维度 (Check if dimensions need padding)
                            for feature in ["observation.state", "action"]:
                                if feature in df.columns:
                                    # 检查第一个非空值 (Check first non-null value)
                                    for _idx, value in enumerate(df[feature]):
                                        if value is not None and isinstance(value, (list, np.ndarray)):
                                            current_dim = len(value)
                                            if current_dim < max_dim:
                                                print(
                                                    f"填充 {feature} 从 {current_dim} 维到 {max_dim} 维 (Padding {feature} from {current_dim} to {max_dim} dimensions)"
                                                )
                                                # 使用零填充到目标维度 (Pad with zeros to target dimension)
                                                df[feature] = df[feature].apply(
                                                    lambda x: np.pad(
                                                        x, (0, max_dim - len(x)), "constant"
                                                    ).tolist()
                                                    if x is not None
                                                    and isinstance(x, (list, np.ndarray))
                                                    and len(x) < max_dim
                                                    else x
                                                )
                                            break

                            # 更新episode_index列 (Update episode_index column)
                            if "episode_index" in df.columns:
                                print(
                                    f"更新episode_index从 {df['episode_index'].iloc[0]} 到 {new_index} (Update episode_index from {df['episode_index'].iloc[0]} to {new_index})"
                                )
                                df["episode_index"] = new_index

                            # 更新index列 (Update index column)
                            if "index" in df.columns:
                                if episode_to_frame_index and new_index in episode_to_frame_index:
                                    # 使用预先计算的帧索引起始值 (Use pre-calculated frame index start value)
                                    first_index = episode_to_frame_index[new_index]
                                    print(
                                        f"更新index列，起始值: {first_index}（使用全局累积帧计数）(Update index column, start value: {first_index} (using global cumulative frame count))"
                                    )
                                else:
                                    # 如果没有提供映射，使用当前的计算方式作为回退
                                    # (If no mapping provided, use current calculation as fallback)
                                    first_index = new_index * len(df)
                                    print(
                                        f"更新index列，起始值: {first_index}（使用episode索引乘以长度）(Update index column, start value: {first_index} (using episode index multiplied by length))"
                                    )

                                # 更新所有帧的索引 (Update indices for all frames)
                                df["index"] = [first_index + i for i in range(len(df))]

                            # 更新task_index列 (Update task_index column)
                            if (
                                "task_index" in df.columns
                                and folder_task_mapping
                                and old_folder in folder_task_mapping
                            ):
                                # 获取当前task_index (Get current task_index)
                                current_task_index = df["task_index"].iloc[0]

                                # 检查是否有对应的新索引 (Check if there's a corresponding new index)
                                if current_task_index in folder_task_mapping[old_folder]:
                                    new_task_index = folder_task_mapping[old_folder][current_task_index]
                                    print(
                                        f"更新task_index从 {current_task_index} 到 {new_task_index} (Update task_index from {current_task_index} to {new_task_index})"
                                    )
                                    df["task_index"] = new_task_index
                                else:
                                    print(
                                        f"警告: 找不到task_index {current_task_index}的映射关系 (Warning: No mapping found for task_index {current_task_index})"
                                    )

                            # 计算chunk编号 (Calculate chunk number)
                            chunk_index = new_index // chunks_size

                            # 创建正确的目标目录 (Create correct target directory)
                            chunk_dir = os.path.join(output_folder, "data", f"chunk-{chunk_index:03d}")
                            os.makedirs(chunk_dir, exist_ok=True)

                            # 构建正确的目标路径 (Build correct target path)
                            dest_path = os.path.join(chunk_dir, f"episode_{new_index:06d}.parquet")

                            # 保存到正确位置 (Save to correct location)
                            df.to_parquet(dest_path, index=False)

                            total_copied += 1
                            found = True
                            print(f"已处理并保存: {dest_path} (Processed and saved: {dest_path})")
                            break
                        except Exception as e:
                            error_msg = f"处理 {source_path} 失败: {e} (Processing {source_path} failed: {e})"
                            print(error_msg)
                            traceback.print_exc()
                            failed_files.append({"file": source_path, "reason": str(e), "episode": old_index})
                            total_failed += 1
                if found:
                    break

            if not found:
                error_msg = f"找不到episode {old_index}的parquet文件，源文件夹: {old_folder}"
                print(error_msg)
                failed_files.append(
                    {"file": f"episode_{old_index:06d}.parquet", "reason": "文件未找到", "folder": old_folder}
                )
                total_failed += 1

    print(f"共复制 {total_copied} 个数据文件，{total_failed} 个失败")

    # 打印所有失败的文件详情 (Print details of all failed files)
    if failed_files:
        print("\n失败的文件详情 (Details of failed files):")
        for i, failed in enumerate(failed_files):
            print(f"{i + 1}. 文件 (File): {failed['file']}")
            if "folder" in failed:
                print(f"   文件夹 (Folder): {failed['folder']}")
            if "episode" in failed:
                print(f"   Episode索引 (Episode index): {failed['episode']}")
            print(f"   原因 (Reason): {failed['reason']}")
            print("---")

    return total_copied > 0


def pad_parquet_data(source_path, target_path, original_dim=14, target_dim=18):
    """
    通过零填充将parquet数据从原始维度扩展到目标维度
    (Extend parquet data from original dimension to target dimension by zero-padding)

    Args:
        source_path (str): 源parquet文件路径 (Source parquet file path)
        target_path (str): 目标parquet文件路径 (Target parquet file path)
        original_dim (int): 原始向量维度 (Original vector dimension)
        target_dim (int): 目标向量维度 (Target vector dimension)
    """
    # 读取parquet文件
    df = pd.read_parquet(source_path)

    # 打印列名以便调试
    print(f"Columns in {source_path}: {df.columns.tolist()}")

    # 创建新的DataFrame来存储填充后的数据
    new_df = df.copy()

    # 检查observation.state和action列是否存在
    if "observation.state" in df.columns:
        # 检查第一行数据，确认是否为向量
        first_state = df["observation.state"].iloc[0]
        print(f"First observation.state type: {type(first_state)}, value: {first_state}")

        # 如果是向量（列表或numpy数组）
        if isinstance(first_state, (list, np.ndarray)):
            # 检查维度
            state_dim = len(first_state)
            print(f"observation.state dimension: {state_dim}")

            if state_dim < target_dim:
                # 填充向量
                print(f"Padding observation.state from {state_dim} to {target_dim} dimensions")
                new_df["observation.state"] = df["observation.state"].apply(
                    lambda x: np.pad(x, (0, target_dim - len(x)), "constant").tolist()
                )

    # 同样处理action列
    if "action" in df.columns:
        # 检查第一行数据
        first_action = df["action"].iloc[0]
        print(f"First action type: {type(first_action)}, value: {first_action}")

        # 如果是向量
        if isinstance(first_action, (list, np.ndarray)):
            # 检查维度
            action_dim = len(first_action)
            print(f"action dimension: {action_dim}")

            if action_dim < target_dim:
                # 填充向量
                print(f"Padding action from {action_dim} to {target_dim} dimensions")
                new_df["action"] = df["action"].apply(
                    lambda x: np.pad(x, (0, target_dim - len(x)), "constant").tolist()
                )

    # 确保目标目录存在
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    # 保存到新的parquet文件
    new_df.to_parquet(target_path, index=False)

    print(f"已将{source_path}处理并保存到{target_path}")

    return new_df


def merge_datasets(
    source_folders, output_folder, validate_ts=False, tolerance_s=1e-4, max_dim=18, default_fps=20
):
    """
    将多个数据集文件夹合并为一个，处理索引、维度和元数据
    (Merge multiple dataset folders into one, handling indices, dimensions, and metadata)

    Args:
        source_folders (list): 源数据集文件夹路径列表 (List of source dataset folder paths)
        output_folder (str): 输出文件夹路径 (Output folder path)
        validate_ts (bool): 是否验证时间戳 (Whether to validate timestamps)
        tolerance_s (float): 时间戳不连续性的容差值，以秒为单位 (Tolerance for timestamp discontinuities in seconds)
        max_dim (int): 向量的最大维度 (Maximum dimension for vectors)
        default_fps (float): 默认帧率 (Default frame rate)

    这个函数执行以下操作:
    (This function performs the following operations:)
    1. 合并所有的episodes、tasks和stats (Merges all episodes, tasks and stats)
    2. 重新编号所有的索引以保持连续性 (Renumbers all indices to maintain continuity)
    3. 填充向量维度使其一致 (Pads vector dimensions for consistency)
    4. 更新元数据文件 (Updates metadata files)
    5. 复制并处理数据和视频文件 (Copies and processes data and video files)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "meta"), exist_ok=True)

    # 注释掉时间戳验证
    # if validate_ts:
    #     issues, fps_values = validate_timestamps(source_folders, tolerance_s)
    #     if issues:
    #         print("时间戳验证发现以下问题:")
    #         for issue in issues:
    #             print(f"  - {issue}")
    #
    #     # 获取共同的FPS值
    #     if fps_values:
    #         fps = max(set(fps_values), key=fps_values.count)
    #         print(f"使用共同FPS值: {fps}")
    #     else:
    #         fps = default_fps
    #         print(f"未找到FPS值，使用默认值: {default_fps}")
    # else:
    fps = default_fps
    print(f"使用默认FPS值: {fps}")

    # Load episodes from all source folders
    all_episodes = []
    all_episodes_stats = []
    all_tasks = []

    total_frames = 0
    total_episodes = 0

    # Keep track of episode mapping (old_folder, old_index, new_index)
    episode_mapping = []

    # Collect all stats for proper merging
    all_stats_data = []

    # Track dimensions for each folder
    folder_dimensions = {}

    # 添加一个变量来跟踪累积的帧数
    cumulative_frame_count = 0

    # 创建一个映射，用于存储每个新的episode索引对应的起始帧索引
    episode_to_frame_index = {}

    # 创建一个映射，用于跟踪旧的任务描述到新任务索引的映射
    task_desc_to_new_index = {}
    # 创建一个映射，用于存储每个源文件夹和旧任务索引到新任务索引的映射
    folder_task_mapping = {}

    # 首先收集所有不同的任务描述
    all_unique_tasks = []

    # 从info.json获取chunks_size
    info_path = os.path.join(source_folders[0], "meta", "info.json")
    chunks_size = 1000  # 默认值
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
            chunks_size = info.get("chunks_size", 1000)

    # 使用更简单的方法计算视频总数 (Use simpler method to calculate total videos)
    total_videos = 0

    for folder in source_folders:
        try:
            # 从每个数据集的info.json直接获取total_videos
            # (Get total_videos directly from each dataset's info.json)
            folder_info_path = os.path.join(folder, "meta", "info.json")
            if os.path.exists(folder_info_path):
                with open(folder_info_path) as f:
                    folder_info = json.load(f)
                    if "total_videos" in folder_info:
                        folder_videos = folder_info["total_videos"]
                        total_videos += folder_videos
                        print(
                            f"从{folder}的info.json中读取到视频数量: {folder_videos} (Read video count from {folder}'s info.json: {folder_videos})"
                        )

            # Check dimensions of state vectors in this folder
            folder_dim = max_dim  # 使用变量替代硬编码的18

            # Try to find a parquet file to determine dimensions
            for root, _dirs, files in os.walk(folder):
                for file in files:
                    if file.endswith(".parquet"):
                        try:
                            df = pd.read_parquet(os.path.join(root, file))
                            if "observation.state" in df.columns:
                                first_state = df["observation.state"].iloc[0]
                                if isinstance(first_state, (list, np.ndarray)):
                                    folder_dim = len(first_state)
                                    print(f"Detected {folder_dim} dimensions in {folder}")
                                    break
                        except Exception as e:
                            print(f"Error checking dimensions in {folder}: {e}")
                        break
                if folder_dim != max_dim:  # 使用变量替代硬编码的18
                    break

            folder_dimensions[folder] = folder_dim

            # Load episodes
            episodes_path = os.path.join(folder, "meta", "episodes.jsonl")
            if not os.path.exists(episodes_path):
                print(f"Warning: Episodes file not found in {folder}, skipping")
                continue

            episodes = load_jsonl(episodes_path)

            # Load episode stats
            episodes_stats_path = os.path.join(folder, "meta", "episodes_stats.jsonl")
            episodes_stats = []
            if os.path.exists(episodes_stats_path):
                episodes_stats = load_jsonl(episodes_stats_path)

            # Create a mapping of episode_index to stats
            stats_map = {}
            for stat in episodes_stats:
                if "episode_index" in stat:
                    stats_map[stat["episode_index"]] = stat

            # Load tasks
            tasks_path = os.path.join(folder, "meta", "tasks.jsonl")
            folder_tasks = []
            if os.path.exists(tasks_path):
                folder_tasks = load_jsonl(tasks_path)

            # 创建此文件夹的任务映射
            folder_task_mapping[folder] = {}

            # 处理每个任务
            for task in folder_tasks:
                task_desc = task["task"]
                old_index = task["task_index"]

                # 检查任务描述是否已存在
                if task_desc not in task_desc_to_new_index:
                    # 添加新任务描述，分配新索引
                    new_index = len(all_unique_tasks)
                    task_desc_to_new_index[task_desc] = new_index
                    all_unique_tasks.append({"task_index": new_index, "task": task_desc})

                # 保存此文件夹中旧索引到新索引的映射
                folder_task_mapping[folder][old_index] = task_desc_to_new_index[task_desc]

            # Process all episodes from this folder
            for episode in episodes:
                old_index = episode["episode_index"]
                new_index = total_episodes

                # Update episode index
                episode["episode_index"] = new_index
                all_episodes.append(episode)

                # Update stats if available
                if old_index in stats_map:
                    stats = stats_map[old_index]
                    stats["episode_index"] = new_index

                    # Pad stats data if needed
                    if "stats" in stats and folder_dimensions[folder] < max_dim:  # 使用变量替代硬编码的18
                        # Pad observation.state and action stats
                        for feature in ["observation.state", "action"]:
                            if feature in stats["stats"]:
                                for stat_type in ["mean", "std", "max", "min"]:
                                    if stat_type in stats["stats"][feature]:
                                        # Get current values
                                        values = stats["stats"][feature][stat_type]

                                        # Check if it's a list/array that needs padding
                                        if (
                                            isinstance(values, list) and len(values) < max_dim
                                        ):  # 使用变量替代硬编码的18
                                            # Pad with zeros
                                            padded = values + [0.0] * (
                                                max_dim - len(values)
                                            )  # 使用变量替代硬编码的18
                                            stats["stats"][feature][stat_type] = padded

                    all_episodes_stats.append(stats)

                    # Add to all_stats_data for proper merging
                    if "stats" in stats:
                        all_stats_data.append(stats["stats"])

                # Add to mapping
                episode_mapping.append((folder, old_index, new_index))

                # Update counters
                total_episodes += 1
                total_frames += episode["length"]

                # 处理每个episode时收集此信息
                episode_to_frame_index[new_index] = cumulative_frame_count
                cumulative_frame_count += episode["length"]

            # 使用收集的唯一任务列表替换之前的任务处理逻辑
            all_tasks = all_unique_tasks

        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
            continue

    print(f"Processed {total_episodes} episodes from {len(source_folders)} folders")

    # Save combined episodes and stats
    save_jsonl(all_episodes, os.path.join(output_folder, "meta", "episodes.jsonl"))
    save_jsonl(all_episodes_stats, os.path.join(output_folder, "meta", "episodes_stats.jsonl"))
    save_jsonl(all_tasks, os.path.join(output_folder, "meta", "tasks.jsonl"))

    # Merge and save stats
    stats_list = []
    for folder in source_folders:
        stats_path = os.path.join(folder, "meta", "stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
                stats_list.append(stats)

    if stats_list:
        # Merge global stats
        merged_stats = merge_stats(stats_list)

        # Update merged stats with episode-specific stats if available
        if all_stats_data:
            # For each feature in the stats
            for feature in merged_stats:
                if feature in all_stats_data[0]:
                    # Recalculate statistics based on all episodes
                    values = [stat[feature] for stat in all_stats_data if feature in stat]

                    # Find the maximum dimension for this feature
                    max_dim = max(
                        len(np.array(val.get("mean", [0])).flatten()) for val in values if "mean" in val
                    )

                    # Update count
                    if "count" in merged_stats[feature]:
                        merged_stats[feature]["count"] = [
                            sum(stat.get("count", [0])[0] for stat in values if "count" in stat)
                        ]

                    # Update min/max with padding
                    if "min" in merged_stats[feature] and all("min" in stat for stat in values):
                        # Pad min values
                        padded_mins = []
                        for val in values:
                            val_array = np.array(val["min"])
                            val_flat = val_array.flatten()
                            if len(val_flat) < max_dim:
                                padded = np.zeros(max_dim)
                                padded[: len(val_flat)] = val_flat
                                padded_mins.append(padded)
                            else:
                                padded_mins.append(val_flat)
                        merged_stats[feature]["min"] = np.minimum.reduce(padded_mins).tolist()

                    if "max" in merged_stats[feature] and all("max" in stat for stat in values):
                        # Pad max values
                        padded_maxs = []
                        for val in values:
                            val_array = np.array(val["max"])
                            val_flat = val_array.flatten()
                            if len(val_flat) < max_dim:
                                padded = np.zeros(max_dim)
                                padded[: len(val_flat)] = val_flat
                                padded_maxs.append(padded)
                            else:
                                padded_maxs.append(val_flat)
                        merged_stats[feature]["max"] = np.maximum.reduce(padded_maxs).tolist()

                    # Update mean and std (weighted by count if available)
                    if "mean" in merged_stats[feature] and all("mean" in stat for stat in values):
                        # Pad mean values
                        padded_means = []
                        for val in values:
                            val_array = np.array(val["mean"])
                            val_flat = val_array.flatten()
                            if len(val_flat) < max_dim:
                                padded = np.zeros(max_dim)
                                padded[: len(val_flat)] = val_flat
                                padded_means.append(padded)
                            else:
                                padded_means.append(val_flat)

                        if all("count" in stat for stat in values):
                            counts = [stat["count"][0] for stat in values]
                            total_count = sum(counts)
                            weighted_means = [
                                mean * count / total_count
                                for mean, count in zip(padded_means, counts, strict=False)
                            ]
                            merged_stats[feature]["mean"] = np.sum(weighted_means, axis=0).tolist()
                        else:
                            merged_stats[feature]["mean"] = np.mean(padded_means, axis=0).tolist()

                    if "std" in merged_stats[feature] and all("std" in stat for stat in values):
                        # Pad std values
                        padded_stds = []
                        for val in values:
                            val_array = np.array(val["std"])
                            val_flat = val_array.flatten()
                            if len(val_flat) < max_dim:
                                padded = np.zeros(max_dim)
                                padded[: len(val_flat)] = val_flat
                                padded_stds.append(padded)
                            else:
                                padded_stds.append(val_flat)

                        if all("count" in stat for stat in values):
                            counts = [stat["count"][0] for stat in values]
                            total_count = sum(counts)
                            variances = [std**2 for std in padded_stds]
                            weighted_variances = [
                                var * count / total_count
                                for var, count in zip(variances, counts, strict=False)
                            ]
                            merged_stats[feature]["std"] = np.sqrt(
                                np.sum(weighted_variances, axis=0)
                            ).tolist()
                        else:
                            # Simple average of standard deviations
                            merged_stats[feature]["std"] = np.mean(padded_stds, axis=0).tolist()

        with open(os.path.join(output_folder, "meta", "stats.json"), "w") as f:
            json.dump(merged_stats, f, indent=4)

    # Update and save info.json
    info_path = os.path.join(source_folders[0], "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)

    # Update info with correct counts
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = len(all_tasks)
    info["total_chunks"] = (total_episodes + info["chunks_size"] - 1) // info[
        "chunks_size"
    ]  # Ceiling division

    # Update splits
    info["splits"] = {"train": f"0:{total_episodes}"}

    # Update feature dimensions to the maximum dimension
    if "features" in info:
        # Find the maximum dimension across all folders
        actual_max_dim = max_dim  # 使用变量替代硬编码的18
        for _folder, dim in folder_dimensions.items():
            actual_max_dim = max(actual_max_dim, dim)

        # Update observation.state and action dimensions
        for feature_name in ["observation.state", "action"]:
            if feature_name in info["features"] and "shape" in info["features"][feature_name]:
                info["features"][feature_name]["shape"] = [actual_max_dim]
                print(f"Updated {feature_name} shape to {actual_max_dim}")

    # 更新视频总数 (Update total videos)
    info["total_videos"] = total_videos
    print(f"更新视频总数为: {total_videos} (Update total videos to: {total_videos})")

    with open(os.path.join(output_folder, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    # Copy video and data files
    copy_videos(source_folders, output_folder, episode_mapping)
    copy_data_files(
        source_folders,
        output_folder,
        episode_mapping,
        max_dim=max_dim,
        fps=fps,
        episode_to_frame_index=episode_to_frame_index,
        folder_task_mapping=folder_task_mapping,
        chunks_size=chunks_size,
    )

    print(f"Merged {total_episodes} episodes with {total_frames} frames into {output_folder}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Merge datasets from multiple sources.")

    # Add arguments
    parser.add_argument("--sources", nargs="+", required=True, help="List of source folder paths")
    parser.add_argument("--output", required=True, help="Output folder path")
    parser.add_argument("--max_dim", type=int, default=32, help="Maximum dimension (default: 32)")
    parser.add_argument("--fps", type=int, default=20, help="Your datasets FPS (default: 20)")

    # Parse arguments
    args = parser.parse_args()

    # Use parsed arguments
    merge_datasets(args.sources, args.output, max_dim=args.max_dim, default_fps=args.fps)
