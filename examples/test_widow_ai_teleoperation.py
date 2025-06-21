#!/usr/bin/env python

"""
Test script to verify that the teleoperation system works with Widow AI components.

This script tests the configuration and factory patterns without connecting to actual hardware.
"""

import sys
from pathlib import Path

# Add the lerobot package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lerobot.common.robots import make_robot_from_config
from lerobot.common.teleoperators import make_teleoperator_from_config
from lerobot.common.robots.widow_ai_follower import WidowAIFollowerConfig
from lerobot.common.teleoperators.widow_ai_leader import WidowAILeaderConfig


def test_configurations():
    """Test that configurations can be created and validated."""
    print("Testing Widow AI configurations...")
    
    # Test follower config
    follower_config = WidowAIFollowerConfig(
        id="test_follower",
        port="192.168.1.102",
        model="V0_FOLLOWER",
        use_degrees=True,
        max_relative_target=30,
    )
    print(f"✓ Follower config created: {follower_config.type}")
    
    # Test leader config
    leader_config = WidowAILeaderConfig(
        id="test_leader",
        port="192.168.1.101",
        model="V0_LEADER",
    )
    print(f"✓ Leader config created: {leader_config.type}")
    
    return follower_config, leader_config


def test_factory_patterns():
    """Test that the factory patterns work correctly."""
    print("\nTesting factory patterns...")
    
    # Test robot factory
    follower_config = WidowAIFollowerConfig(
        id="test_follower",
        port="192.168.1.102",
        model="V0_FOLLOWER",
    )
    
    try:
        follower = make_robot_from_config(follower_config)
        print(f"✓ Robot factory works: {follower.__class__.__name__}")
        print(f"  - Action features: {list(follower.action_features.keys())}")
        print(f"  - Observation features: {list(follower.observation_features.keys())}")
    except Exception as e:
        print(f"✗ Robot factory failed: {e}")
        return False
    
    # Test teleoperator factory
    leader_config = WidowAILeaderConfig(
        id="test_leader",
        port="192.168.1.101",
        model="V0_LEADER",
    )
    
    try:
        leader = make_teleoperator_from_config(leader_config)
        print(f"✓ Teleoperator factory works: {leader.__class__.__name__}")
        print(f"  - Action features: {list(leader.action_features.keys())}")
        print(f"  - Feedback features: {list(leader.feedback_features.keys())}")
    except Exception as e:
        print(f"✗ Teleoperator factory failed: {e}")
        return False
    
    return True


def test_teleoperation_compatibility():
    """Test that the components are compatible with the teleoperation system."""
    print("\nTesting teleoperation compatibility...")
    
    # Test that action features match
    follower_config = WidowAIFollowerConfig(
        id="test_follower",
        port="192.168.1.102",
        model="V0_FOLLOWER",
    )
    
    leader_config = WidowAILeaderConfig(
        id="test_leader",
        port="192.168.1.101",
        model="V0_LEADER",
    )
    
    follower = make_robot_from_config(follower_config)
    leader = make_teleoperator_from_config(leader_config)
    
    # Check that action features are compatible
    leader_actions = set(leader.action_features.keys())
    follower_actions = set(follower.action_features.keys())
    
    if leader_actions == follower_actions:
        print("✓ Action features are compatible")
        print(f"  - Actions: {sorted(leader_actions)}")
    else:
        print("✗ Action features are incompatible")
        print(f"  - Leader actions: {sorted(leader_actions)}")
        print(f"  - Follower actions: {sorted(follower_actions)}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("Testing Widow AI teleoperation integration...")
    print("=" * 50)
    
    try:
        # Test configurations
        test_configurations()
        
        # Test factory patterns
        if not test_factory_patterns():
            print("\n✗ Factory pattern tests failed")
            return False
        
        # Test teleoperation compatibility
        if not test_teleoperation_compatibility():
            print("\n✗ Teleoperation compatibility tests failed")
            return False
        
        print("\n" + "=" * 50)
        print("✓ All tests passed! The Widow AI components are ready for teleoperation.")
        print("\nYou can now use the teleoperation system with:")
        print("python -m lerobot.teleoperate \\")
        print("    --robot.type=widow_ai_follower \\")
        print("    --robot.port=192.168.1.102 \\")
        print("    --robot.id=follower_1 \\")
        print("    --teleop.type=widow_ai_leader \\")
        print("    --teleop.port=192.168.1.101 \\")
        print("    --teleop.id=leader_1 \\")
        print("    --fps=10")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 