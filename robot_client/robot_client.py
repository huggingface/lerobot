import logging
import io
import pickle  # nosec
import threading
import time
from dataclasses import asdict
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import torch

# Импорт с поддержкой запуска как пакета и как скрипта
try:
    from robot_client.configs import RobotClientConfig
    from robot_client.dummy_robot import DummyRobot
    from robot_client.features import map_robot_keys_to_lerobot_features
    from robot_client.support import (
        FPSTracker,
        RemotePolicyConfig,
        TimedAction,
        TimedObservation,
        get_logger,
    )
    from robot_client.transport import services_pb2, services_pb2_grpc
    from robot_client.transport.utils import grpc_channel_options, send_bytes_in_chunks
except Exception:  # noqa: BLE001
    from configs import RobotClientConfig  # type: ignore
    from dummy_robot import DummyRobot  # type: ignore
    from features import map_robot_keys_to_lerobot_features  # type: ignore
    from support import (  # type: ignore
        FPSTracker,
        RemotePolicyConfig,
        TimedAction,
        TimedObservation,
        get_logger,
    )
    from transport import services_pb2, services_pb2_grpc  # type: ignore
    from transport.utils import grpc_channel_options, send_bytes_in_chunks  # type: ignore

# Шим для совместимости pickle с сервером: подменяем модуль
# 'lerobot.async_inference.helpers' на локальные классы, чтобы
# уметь распаковывать TimedAction/TimedObservation и корректно
# сериализовать RemotePolicyConfig.
import sys
import types

shim_mod_name = "lerobot.async_inference.helpers"
if shim_mod_name not in sys.modules:
    try:
        # Найти локальный модуль с классами
        # (после блока импорта это либо robot_client.support, либо support)
        support_module = sys.modules.get("robot_client.support") or sys.modules.get("support")
        if support_module is not None:
            shim = types.ModuleType(shim_mod_name)
            for _name in ("TimedAction", "TimedObservation", "RemotePolicyConfig"):
                if hasattr(support_module, _name):
                    setattr(shim, _name, getattr(support_module, _name))
            sys.modules[shim_mod_name] = shim
    except Exception:  # noqa: BLE001
        pass

# Обновляем модульные имена только для TimedAction/TimedObservation
try:
    for _cls in (TimedAction, TimedObservation):
        _cls.__module__ = "lerobot.async_inference.helpers"
except Exception:  # noqa: BLE001
    pass


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        self.config = config
        self.robot = DummyRobot(
            robot_id=config.robot_id,
            num_joints=config.num_joints,
            cameras=config.cameras,
            action_extra_names=config.extra_actions,
            fps=config.fps,
        )
        self.robot.connect()

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
        )
        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()

        # client vars
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)

        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

        self.must_go = threading.Event()
        self.must_go.set()

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self):
        try:
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # Отправляем словарь вместо класса, чтобы не требовать импортов на сервере
            policy_config_bytes = pickle.dumps(asdict(self.policy_config))
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)
            self.shutdown_event.clear()
            return True
        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        self.shutdown_event.set()
        self.robot.disconnect()
        self.logger.debug("Robot disconnected")
        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    def send_observation(self, obs: TimedObservation) -> bool:
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")
        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        # Отправляем простой словарь, чтобы избежать требований к модульным путям классов
        obs_payload = {
            "timestamp": obs.get_timestamp(),
            "timestep": obs.get_timestep(),
            "observation": obs.get_observation(),
            "must_go": obs.must_go,
        }
        observation_bytes = pickle.dumps(obs_payload)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.debug(f"Sent observation #{obs_timestep} | ")
            return True
        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _aggregate_action_queues(self, incoming_actions: list[TimedAction], aggregate_fn):
        if aggregate_fn is None:
            def aggregate_fn(x1, x2):  # noqa: ANN001
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action
            if new_action.get_timestep() <= latest_action:
                continue
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(current_action_queue[new_action.get_timestep()], new_action.get_action()),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self, verbose: bool = False):
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue

                receive_time = time.time()
                deserialize_start = time.perf_counter()
                # Безопасная десериализация действий на CPU (даже если сервер прислал CUDA-тензоры)
                try:
                    timed_actions = torch.load(io.BytesIO(actions_chunk.data), map_location="cpu", weights_only=False)
                except Exception:
                    # Fallback: обычный pickle (если объект не содержит тензоры/стораджи)
                    timed_actions = pickle.loads(actions_chunk.data)  # nosec
                # Interop: если получили список словарей, восстановим TimedAction локально
                if isinstance(timed_actions, list) and len(timed_actions) > 0 and isinstance(timed_actions[0], dict):
                    restored: list[TimedAction] = []
                    for item in timed_actions:
                        action_tensor = torch.tensor(item["action"], dtype=torch.float32)
                        restored.append(
                            TimedAction(
                                timestamp=float(item["timestamp"]),
                                timestep=int(item["timestep"]),
                                action=action_tensor,
                            )
                        )
                    timed_actions = restored
                deserialize_time = time.perf_counter() - deserialize_start
                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Явное сообщение о получении нового чанка действий
                try:
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]
                    if len(incoming_timesteps) > 0:
                        self.logger.info(
                            f"[ACTIONS] Received action chunk: steps {incoming_timesteps[0]}..{incoming_timesteps[-1]} "
                            f"(size={len(incoming_timesteps)})"
                        )
                except Exception:  # noqa: BLE001
                    pass

                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action
                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000
                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time
                self.must_go.set()

                if verbose:
                    with self.action_queue_lock:
                        new_size = self.action_queue.qsize()
                    with self.latest_action_lock:
                        latest_action = self.latest_action
                    self.logger.debug(
                        f"Latest action: {latest_action} | "
                        f"Queue size after update: {new_size} | "
                        f"Queue update time: {queue_update_time:.6f}s"
                    )
            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")
            except Exception as e:  # noqa: BLE001
                self.logger.error(f"Unexpected error in receive_actions: {e}")

    def actions_available(self):
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        return {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        _performed_action = self.robot.send_action(self._action_tensor_to_action_dict(timed_action.get_action()))
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        # Явное сообщение об извлечении действия из очереди (из полученного ранее чанка)
        self.logger.info(f"[ACTIONS] Dequeued action step {timed_action.get_timestep()} from queue")

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()
            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )
            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )
        return _performed_action

    def _ready_to_send_observation(self):
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False):
        try:
            start_time = time.perf_counter()
            raw_observation = self.robot.get_observation()
            raw_observation["task"] = task

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()
            _ = self.send_observation(observation)

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                self.must_go.clear()

            if verbose:
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())
                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )
                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )
            return raw_observation
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Error in observation sender: {e}")

    def control_loop(self, task: str, verbose: bool = False):
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")
        _performed_action = None
        _captured_observation = None
        while self.running:
            control_loop_start = time.perf_counter()
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)
            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(task, verbose)
            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))
        return _captured_observation, _performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))
    client = RobotClient(cfg)
    if client.start():
        client.logger.info("Starting action receiver thread...")
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()
        try:
            client.control_loop(task=cfg.task)
        finally:
            client.stop()
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                try:
                    import matplotlib.pyplot as plt  # lazy import
                    plt.figure()
                    plt.title("Action Queue Size Over Time")
                    plt.plot(range(len(client.action_queue_size)), client.action_queue_size)
                    plt.xlabel("Environment steps")
                    plt.ylabel("Queue size")
                    plt.grid(True, alpha=0.3)
                    plt.show()
                except Exception:  # noqa: BLE001
                    pass
            client.logger.info("Client stopped")


if __name__ == "__main__":
    async_client()
