"""
LeRobot Integration Guide for Control Loop Tracking

This file shows how to integrate the tracking decorators with actual LeRobot components
for comprehensive monitoring of control loops and method calls.
"""

import logging

from control_loop_trackers import (
    MethodTracker,
    track_control_loop,
    track_method,
)


def setup_robot_client_tracking():
    """
    Set up tracking for RobotClient control loops.
    Call this before starting your robot client.
    """
    try:
        from lerobot.scripts.server.robot_client import RobotClient

        # Create dedicated tracker for robot client
        robot_tracker = MethodTracker("robot_client")

        # Track the main control loop with FPS monitoring
        original_control_loop = RobotClient.control_loop
        RobotClient.control_loop = track_control_loop(
            tracker=robot_tracker,
            target_fps=30.0,
            warn_slow_methods=True,
            slow_threshold_ms=33.0,  # 30Hz = 33ms per loop
        )(original_control_loop)

        # Track key methods in the control loop
        RobotClient.apply_action = track_method(tracker=robot_tracker, log_calls=False, log_timing=True)(
            RobotClient.apply_action
        )

        RobotClient._update_action_queue = track_method(
            tracker=robot_tracker, log_calls=False, log_timing=True
        )(RobotClient._update_action_queue)

        RobotClient.push_observation_to_queue = track_method(
            tracker=robot_tracker, log_calls=False, log_timing=True
        )(RobotClient.push_observation_to_queue)

        RobotClient.track_action_queue_size = track_method(
            tracker=robot_tracker, log_calls=False, log_timing=True
        )(RobotClient.track_action_queue_size)

        logging.info("✅ Robot client tracking enabled")
        return robot_tracker

    except ImportError:
        logging.warning("⚠️  Could not import RobotClient - tracking not enabled")
        return None


def setup_policy_tracking():
    """
    Set up tracking for policy inference and prediction.
    """
    try:
        from lerobot.utils.control_utils import predict_action

        # Create tracker for policy operations
        policy_tracker = MethodTracker("policy_inference")

        # Track the main prediction function
        original_predict_action = predict_action

        @track_method(tracker=policy_tracker, log_calls=True, log_timing=True, log_args=False)
        def tracked_predict_action(*args, **kwargs):
            return original_predict_action(*args, **kwargs)

        # Replace the function globally
        import lerobot.utils.control_utils

        lerobot.utils.control_utils.predict_action = tracked_predict_action

        logging.info("✅ Policy prediction tracking enabled")
        return policy_tracker

    except ImportError:
        logging.warning("⚠️  Could not import predict_action - tracking not enabled")
        return None


def setup_recording_tracking():
    """
    Set up tracking for data recording loops.
    """
    try:
        from lerobot.record import record_loop

        # Create tracker for recording
        record_tracker = MethodTracker("recording")

        # Track the recording loop
        original_record_loop = record_loop

        @track_control_loop(
            tracker=record_tracker, target_fps=30.0, warn_slow_methods=True, slow_threshold_ms=40.0
        )
        def tracked_record_loop(*args, **kwargs):
            return original_record_loop(*args, **kwargs)

        # Replace globally
        import lerobot.record

        lerobot.record.record_loop = tracked_record_loop

        logging.info("✅ Recording loop tracking enabled")
        return record_tracker

    except ImportError:
        logging.warning("⚠️  Could not import record_loop - tracking not enabled")
        return None


def setup_teleop_tracking():
    """
    Set up tracking for teleoperation loops.
    """
    try:
        from lerobot.teleoperate import teleop_loop

        # Create tracker for teleoperation
        teleop_tracker = MethodTracker("teleoperation")

        # Track the teleop loop
        original_teleop_loop = teleop_loop

        @track_control_loop(
            tracker=teleop_tracker,
            target_fps=60.0,  # Teleop usually runs faster
            warn_slow_methods=True,
            slow_threshold_ms=16.0,  # 60Hz = 16ms per loop
        )
        def tracked_teleop_loop(*args, **kwargs):
            return original_teleop_loop(*args, **kwargs)

        # Replace globally
        import lerobot.teleoperate

        lerobot.teleoperate.teleop_loop = tracked_teleop_loop

        logging.info("✅ Teleoperation tracking enabled")
        return teleop_tracker

    except ImportError:
        logging.warning("⚠️  Could not import teleop_loop - tracking not enabled")
        return None


def setup_robot_hardware_tracking():
    """
    Set up tracking for robot hardware interfaces.
    """
    trackers = {}

    # Track SO100 follower robot
    try:
        from lerobot.robots.so100_follower.so100_follower import SO100Follower

        so100_tracker = MethodTracker("so100_hardware")

        # Track key hardware methods
        SO100Follower.get_observation = track_method(tracker=so100_tracker, log_calls=False, log_timing=True)(
            SO100Follower.get_observation
        )

        SO100Follower.send_action = track_method(tracker=so100_tracker, log_calls=False, log_timing=True)(
            SO100Follower.send_action
        )

        trackers["so100"] = so100_tracker
        logging.info("✅ SO100 robot tracking enabled")

    except ImportError:
        logging.warning("⚠️  Could not import SO100Follower")

    # Track SO101 follower robot
    try:
        from lerobot.robots.so101_follower.so101_follower import SO101Follower

        so101_tracker = MethodTracker("so101_hardware")

        SO101Follower.get_observation = track_method(tracker=so101_tracker, log_calls=False, log_timing=True)(
            SO101Follower.get_observation
        )

        SO101Follower.send_action = track_method(tracker=so101_tracker, log_calls=False, log_timing=True)(
            SO101Follower.send_action
        )

        trackers["so101"] = so101_tracker
        logging.info("✅ SO101 robot tracking enabled")

    except ImportError:
        logging.warning("⚠️  Could not import SO101Follower")

    return trackers


def setup_policy_server_tracking():
    """
    Set up tracking for policy server operations.
    """
    try:
        from lerobot.scripts.server.policy_server import PolicyServer

        policy_server_tracker = MethodTracker("policy_server")

        # Track prediction methods
        PolicyServer._predict_action_chunk = track_method(
            tracker=policy_server_tracker, log_calls=True, log_timing=True
        )(PolicyServer._predict_action_chunk)

        PolicyServer._prepare_observation = track_method(
            tracker=policy_server_tracker, log_calls=False, log_timing=True
        )(PolicyServer._prepare_observation)

        PolicyServer._get_action_chunk = track_method(
            tracker=policy_server_tracker, log_calls=False, log_timing=True
        )(PolicyServer._get_action_chunk)

        logging.info("✅ Policy server tracking enabled")
        return policy_server_tracker

    except ImportError:
        logging.warning("⚠️  Could not import PolicyServer")
        return None


def setup_comprehensive_tracking():
    """
    Set up comprehensive tracking across all LeRobot components.
    Returns a dictionary of all trackers for analysis.
    """
    logging.info("🚀 Setting up comprehensive LeRobot tracking...")

    trackers = {}

    # Set up tracking for different components
    robot_client_tracker = setup_robot_client_tracking()
    if robot_client_tracker:
        trackers["robot_client"] = robot_client_tracker

    policy_tracker = setup_policy_tracking()
    if policy_tracker:
        trackers["policy"] = policy_tracker

    recording_tracker = setup_recording_tracking()
    if recording_tracker:
        trackers["recording"] = recording_tracker

    teleop_tracker = setup_teleop_tracking()
    if teleop_tracker:
        trackers["teleop"] = teleop_tracker

    hardware_trackers = setup_robot_hardware_tracking()
    trackers.update(hardware_trackers)

    policy_server_tracker = setup_policy_server_tracking()
    if policy_server_tracker:
        trackers["policy_server"] = policy_server_tracker

    logging.info(f"✅ Comprehensive tracking setup complete. {len(trackers)} trackers active.")
    return trackers


def analyze_all_trackers(trackers: dict):
    """
    Analyze performance data from all active trackers.
    """
    print("\n" + "=" * 80)
    print("🔍 COMPREHENSIVE LEROBOT PERFORMANCE ANALYSIS")
    print("=" * 80)

    total_methods = 0
    total_calls = 0

    for tracker_name, tracker in trackers.items():
        stats = tracker.get_stats()
        if not stats:
            continue

        print(f"\n📊 {tracker_name.upper()} TRACKER:")
        print("-" * 50)

        tracker_calls = 0
        for method_name, stat in stats.items():
            short_name = method_name.split(".")[-1]
            print(
                f"{short_name:25s}: {stat['count']:4d} calls, {stat['avg_time_ms']:6.2f}ms avg, {stat['total_time_ms']:8.1f}ms total"
            )
            tracker_calls += stat["count"]
            total_methods += 1

        total_calls += tracker_calls
        print(f"{'SUBTOTAL':25s}: {tracker_calls:4d} calls")

    print(f"\n{'=' * 50}")
    print(f"{'GRAND TOTAL':25s}: {total_calls:4d} calls across {total_methods} methods")
    print(f"{'=' * 50}")


def export_all_tracking_data(trackers: dict, base_filename: str = "lerobot_tracking"):
    """
    Export tracking data from all trackers to files.
    """
    import time

    timestamp = int(time.time())

    for tracker_name, tracker in trackers.items():
        filename = f"{base_filename}_{tracker_name}_{timestamp}.json"
        tracker.export_to_json(filename)
        print(f"📁 Exported {tracker_name} data to {filename}")


# Example usage patterns
# ======================


def monitor_robot_session():
    """
    Example of monitoring a complete robot session with comprehensive tracking.
    """
    # Set up all tracking
    trackers = setup_comprehensive_tracking()

    try:
        print("🤖 Robot session monitoring active...")
        print("Perform your robot operations now...")
        print("Press Ctrl+C when done.\n")

        # Your robot operations would go here
        # For example:
        # - Start robot client
        # - Run control loops
        # - Record demonstrations
        # - Run policy inference

        import time

        while True:
            time.sleep(1)  # Keep the monitoring active

    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped. Analyzing results...")
        analyze_all_trackers(trackers)
        export_all_tracking_data(trackers)


def debug_control_loop_performance():
    """
    Example of using tracking for debugging control loop performance issues.
    """
    # Set up targeted tracking
    robot_tracker = setup_robot_client_tracking()
    policy_tracker = setup_policy_tracking()

    if not robot_tracker or not policy_tracker:
        print("❌ Could not set up tracking for debugging")
        return

    print("🔧 Debug mode: Tracking control loop performance...")
    print("Run your control loop and look for performance warnings.")

    # The tracking decorators will automatically warn about:
    # - FPS drops below target
    # - Individual methods taking too long
    # - Exceptions in tracked methods

    # After your debugging session:
    def print_debug_summary():
        print("\n🔍 DEBUG SUMMARY:")

        robot_stats = robot_tracker.get_stats()
        policy_stats = policy_tracker.get_stats()

        # Find slowest methods
        all_methods = []
        for stats in [robot_stats, policy_stats]:
            for method, stat in stats.items():
                all_methods.append((method, stat["avg_time_ms"]))

        all_methods.sort(key=lambda x: x[1], reverse=True)

        print("⚠️  SLOWEST METHODS:")
        for i, (method, avg_time) in enumerate(all_methods[:5]):
            print(f"  {i + 1}. {method.split('.')[-1]}: {avg_time:.2f}ms")

    return print_debug_summary


if __name__ == "__main__":
    # Demo the tracking setup
    print("🎯 LeRobot Tracking Integration Demo")
    trackers = setup_comprehensive_tracking()
    print(f"\nActive trackers: {list(trackers.keys())}")
    print("\nNow run your LeRobot applications to see tracking in action!")
