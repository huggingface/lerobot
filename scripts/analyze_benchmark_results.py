#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to analyze benchmark results and generate comparison reports.

This script processes benchmark JSON files and generates detailed analysis
of performance metrics and bottlenecks.
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import seaborn as sns


def load_benchmark_results(file_path: Path) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def analyze_single_benchmark(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single benchmark result."""
    timing_stats = results['timing_stats']
    summary = results['summary']
    
    analysis = {
        'summary': summary,
        'bottlenecks': {},
        'efficiency': {},
        'recommendations': []
    }
    
    # Calculate bottlenecks
    total_time = summary['total_duration']
    categories = ['frame_capture', 'frame_processing', 'image_writing', 'episode_saving', 'video_encoding']
    
    for category in categories:
        if category in timing_stats:
            stats = timing_stats[category]
            if stats['count'] > 0:
                percentage = (stats['total_time'] / total_time) * 100
                analysis['bottlenecks'][category] = {
                    'total_time': stats['total_time'],
                    'percentage': percentage,
                    'mean_time': stats['mean_time'],
                    'count': stats['count']
                }
    
    # Identify primary bottleneck
    if analysis['bottlenecks']:
        primary_bottleneck = max(analysis['bottlenecks'].items(), key=lambda x: x[1]['total_time'])
        analysis['primary_bottleneck'] = primary_bottleneck[0]
    
    # Calculate efficiency metrics
    total_frames = summary['total_frames']
    target_fps = 30  # Assuming 30 FPS target
    
    analysis['efficiency'] = {
        'actual_fps': summary['average_fps'],
        'target_fps': target_fps,
        'fps_efficiency': (summary['average_fps'] / target_fps) * 100,
        'frames_per_episode': total_frames / summary['total_episodes'],
        'time_per_frame': total_time / total_frames
    }
    
    # Generate recommendations
    if analysis['efficiency']['fps_efficiency'] < 90:
        analysis['recommendations'].append("FPS efficiency below 90% - consider optimization")
    
    if 'video_encoding' in analysis['bottlenecks']:
        encoding_percentage = analysis['bottlenecks']['video_encoding']['percentage']
        if encoding_percentage > 30:
            analysis['recommendations'].append(f"Video encoding is {encoding_percentage:.1f}% of total time - consider async encoding")
    
    if 'image_writing' in analysis['bottlenecks']:
        writing_percentage = analysis['bottlenecks']['image_writing']['percentage']
        if writing_percentage > 20:
            analysis['recommendations'].append(f"Image writing is {writing_percentage:.1f}% of total time - consider more threads/processes")
    
    return analysis


def compare_benchmarks(baseline_path: Path, optimized_path: Path) -> Dict[str, Any]:
    """Compare two benchmark results."""
    baseline_results = load_benchmark_results(baseline_path)
    optimized_results = load_benchmark_results(optimized_path)
    
    baseline_analysis = analyze_single_benchmark(baseline_results)
    optimized_analysis = analyze_single_benchmark(optimized_results)
    
    comparison = {
        'baseline': baseline_analysis,
        'optimized': optimized_analysis,
        'improvements': {},
        'regressions': {}
    }
    
    # Compare timing categories
    categories = ['frame_capture', 'frame_processing', 'image_writing', 'episode_saving', 'video_encoding']
    
    for category in categories:
        baseline_time = baseline_analysis['bottlenecks'].get(category, {}).get('total_time', 0)
        optimized_time = optimized_analysis['bottlenecks'].get(category, {}).get('total_time', 0)
        
        if baseline_time > 0 and optimized_time > 0:
            improvement = ((baseline_time - optimized_time) / baseline_time) * 100
            if improvement > 0:
                comparison['improvements'][category] = improvement
            elif improvement < 0:
                comparison['regressions'][category] = abs(improvement)
    
    # Overall improvement
    baseline_total = baseline_analysis['summary']['total_duration']
    optimized_total = optimized_analysis['summary']['total_duration']
    overall_improvement = ((baseline_total - optimized_total) / baseline_total) * 100
    comparison['overall_improvement'] = overall_improvement
    
    return comparison


def create_visualizations(results: Dict[str, Any], output_dir: Path) -> None:
    """Create visualization plots for benchmark results."""
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Timing breakdown pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    
    timing_stats = results['timing_stats']
    categories = []
    times = []
    
    for category, stats in timing_stats.items():
        if stats['count'] > 0:
            categories.append(category.replace('_', ' ').title())
            times.append(stats['total_time'])
    
    if times:
        wedges, texts, autotexts = ax.pie(times, labels=categories, autopct='%1.1f%%', startangle=90)
        ax.set_title('Recording Time Breakdown', fontsize=16, fontweight='bold')
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'timing_breakdown.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Frame timing distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    categories = ['frame_capture', 'frame_processing', 'image_writing', 'video_encoding']
    titles = ['Frame Capture', 'Frame Processing', 'Image Writing', 'Video Encoding']
    
    for i, (category, title) in enumerate(zip(categories, titles)):
        if category in timing_stats and timing_stats[category]['count'] > 0:
            times = timing_stats[category]['times']
            if times:
                axes[i].hist(times, bins=50, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{title} Time Distribution', fontweight='bold')
                axes[i].set_xlabel('Time (seconds)')
                axes[i].set_ylabel('Frequency')
                axes[i].axvline(np.mean(times), color='red', linestyle='--', label=f'Mean: {np.mean(times)*1000:.2f}ms')
                axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'timing_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Episode timeline
    episode_stats = results['episode_stats']
    if episode_stats:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = list(episode_stats.keys())
        durations = [episode_stats[ep]['duration'] for ep in episodes]
        fps_values = [episode_stats[ep]['fps'] for ep in episodes]
        
        x = np.arange(len(episodes))
        width = 0.35
        
        ax.bar(x - width/2, durations, width, label='Duration (s)', alpha=0.8)
        ax_twin = ax.twinx()
        ax_twin.bar(x + width/2, fps_values, width, label='FPS', alpha=0.8, color='orange')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Duration (seconds)')
        ax_twin.set_ylabel('FPS')
        ax.set_title('Episode Performance Timeline', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Ep {ep}' for ep in episodes])
        
        # Add legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'episode_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()


def create_comparison_visualization(comparison: Dict[str, Any], output_dir: Path) -> None:
    """Create comparison visualization between baseline and optimized results."""
    output_dir.mkdir(exist_ok=True)
    
    # 1. Before vs After comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Baseline
    baseline_bottlenecks = comparison['baseline']['bottlenecks']
    categories = list(baseline_bottlenecks.keys())
    baseline_percentages = [baseline_bottlenecks[cat]['percentage'] for cat in categories]
    
    axes[0].pie(baseline_percentages, labels=[cat.replace('_', ' ').title() for cat in categories], 
                autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Baseline Performance', fontweight='bold')
    
    # Optimized
    optimized_bottlenecks = comparison['optimized']['bottlenecks']
    optimized_percentages = [optimized_bottlenecks.get(cat, {}).get('percentage', 0) for cat in categories]
    
    axes[1].pie(optimized_percentages, labels=[cat.replace('_', ' ').title() for cat in categories], 
                autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Optimized Performance', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'before_after_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Improvement chart
    if comparison['improvements']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = list(comparison['improvements'].keys())
        improvements = list(comparison['improvements'].values())
        
        bars = ax.bar(categories, improvements, color='green', alpha=0.7)
        ax.set_title('Performance Improvements', fontweight='bold')
        ax.set_ylabel('Improvement (%)')
        ax.set_xlabel('Component')
        
        # Add value labels on bars
        for bar, value in zip(bars, improvements):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'improvements.png', dpi=300, bbox_inches='tight')
        plt.close()


def generate_report(analysis: Dict[str, Any], output_path: Path) -> None:
    """Generate a detailed text report."""
    with open(output_path, 'w') as f:
        f.write("RECORDING BENCHMARK ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Summary
        summary = analysis['summary']
        f.write("SUMMARY:\n")
        f.write(f"  Total Episodes: {summary['total_episodes']}\n")
        f.write(f"  Total Frames: {summary['total_frames']}\n")
        f.write(f"  Total Duration: {summary['total_duration']:.2f}s\n")
        f.write(f"  Average FPS: {summary['average_fps']:.2f}\n\n")
        
        # Bottlenecks
        f.write("BOTTLENECK ANALYSIS:\n")
        f.write("-" * 20 + "\n")
        for category, stats in analysis['bottlenecks'].items():
            f.write(f"  {category.replace('_', ' ').title()}:\n")
            f.write(f"    Total Time: {stats['total_time']:.2f}s ({stats['percentage']:.1f}%)\n")
            f.write(f"    Mean Time: {stats['mean_time']*1000:.2f}ms\n")
            f.write(f"    Count: {stats['count']}\n\n")
        
        # Efficiency
        efficiency = analysis['efficiency']
        f.write("EFFICIENCY METRICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"  FPS Efficiency: {efficiency['fps_efficiency']:.1f}%\n")
        f.write(f"  Time per Frame: {efficiency['time_per_frame']*1000:.2f}ms\n")
        f.write(f"  Frames per Episode: {efficiency['frames_per_episode']:.0f}\n\n")
        
        # Recommendations
        if analysis['recommendations']:
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 20 + "\n")
            for rec in analysis['recommendations']:
                f.write(f"  • {rec}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("input", type=str, help="Input benchmark JSON file or directory")
    parser.add_argument("--output", type=str, default="./analysis_results", help="Output directory")
    parser.add_argument("--compare", type=str, help="Second benchmark file for comparison")
    parser.add_argument("--format", choices=["json", "csv", "html"], default="json", help="Output format")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)
    
    if input_path.is_file():
        # Single file analysis
        results = load_benchmark_results(input_path)
        analysis = analyze_single_benchmark(results)
        
        # Save analysis
        with open(output_path / "analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Generate report
        generate_report(analysis, output_path / "report.txt")
        
        # Create visualizations
        create_visualizations(results, output_path)
        
        print(f"Analysis completed. Results saved to {output_path}")
        
    elif input_path.is_dir():
        # Directory analysis - find all JSON files
        json_files = list(input_path.glob("*.json"))
        
        if args.compare:
            # Comparison mode
            baseline_file = json_files[0]  # Use first file as baseline
            optimized_file = Path(args.compare)
            
            if baseline_file.exists() and optimized_file.exists():
                comparison = compare_benchmarks(baseline_file, optimized_file)
                
                # Save comparison
                with open(output_path / "comparison.json", 'w') as f:
                    json.dump(comparison, f, indent=2)
                
                # Create comparison visualizations
                create_comparison_visualization(comparison, output_path)
                
                print(f"Comparison completed. Results saved to {output_path}")
            else:
                print("Error: Could not find files for comparison")
        else:
            # Analyze all files
            all_analyses = {}
            for json_file in json_files:
                results = load_benchmark_results(json_file)
                analysis = analyze_single_benchmark(results)
                all_analyses[json_file.stem] = analysis
            
            # Save combined analysis
            with open(output_path / "combined_analysis.json", 'w') as f:
                json.dump(all_analyses, f, indent=2)
            
            print(f"Combined analysis completed. Results saved to {output_path}")


if __name__ == "__main__":
    main() 