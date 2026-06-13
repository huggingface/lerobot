#!/usr/bin/env python

import time
from pynput import keyboard

# Imports LeRobot Core
from lerobot.configs import PreTrainedConfig
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.rollout import RolloutConfig, SentryStrategyConfig, build_rollout_context
from lerobot.rollout.inference import SyncInferenceConfig
from lerobot.rollout.strategies import BaseStrategy, safe_push_to_hub, send_next_action
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging

# Imports Spécifiques Robot & Caméras
from lerobot.cameras.opencv import OpenCVCameraConfig
# Importe directement la Config de ton robot custom
from lerobot.robots.so_follower.so_follower_dragontactile import SO101FollowerDragontactileConfig
#import rerun
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

HF_USERNAME = "jogarulfop"
DATASET_NAME = "my_dataset" # name without rollout_.... The full repo_id will be f"{HF_USERNAME}/rollout_{DATASET_NAME}"
HF_MODEL_ID = "emmanuel-v/policy_2026-05-13_chakeitup_alubox_d"
MY_ROBOT_PORT = "/dev/ttyACM0" 
FPS = 30
SAVE_DATASET_ON_TEARDOWN = False


# --- 1. Définition de ta stratégie personnalisée ---

class ControlledStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.state = "WAITING"
        self.listener = None
        self.home_action = None
        self.reset_started_at = None
        self.reset_hold_seconds = 1.5
        self.rerecord_requested = False
        self._shutdown_event = None
        self._dataset = None
        self._save_dataset_on_teardown = SAVE_DATASET_ON_TEARDOWN
        self.push_requested = False
        self.episode_index = 0

        print("=== Contrôles de la session ===")
        print("[ D ] -> Démarrer l'épisode (Action !)")
        print("[ F ] -> Finir l'épisode (Retour à la base)")
        print("[ R ] -> Refaire l'épisode en cours")
        print("[ ESC ] -> Quitter proprement")
        print("===============================")

    def setup(self, ctx):
        super().setup(ctx)
        self.home_action = dict(ctx.hardware.initial_position or {})
        if not self.home_action:
            raise RuntimeError("Initial robot position is unavailable; cannot build a home action")
        self._shutdown_event = ctx.runtime.shutdown_event
        self._dataset = ctx.data.dataset
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        self._ctx = ctx

    def on_press(self, key):
        try:
            if key.char == 'd':
                if self.state == "WAITING":
                    self.episode_index += 1
                    print(f"\n▶ DÉBUT DE L'ÉPISODE #{self.episode_index} ! Le robot prend le relais.")
                    self.state = "RECORDING"
                return

            elif key.char == 'f':
                if self.state == "RECORDING":
                    print(f"\n⏹ FIN DE L'ÉPISODE {self.episode_index}. Retour à la position de base...")
                    self.rerecord_requested = False
                    self.state = "RESETTING"
                    self.reset_started_at = time.perf_counter()
                return

            elif key.char == 'r':
                if self.state in ("RECORDING", "RESETTING", "WAITING"):
                    print(f"\n↺ REFAIRE L'ÉPISODE {self.episode_index}. Retour à la base puis redémarrage...")
                    self.rerecord_requested = True
                    self.state = "RESETTING"
                    self.reset_started_at = time.perf_counter()
                return

            elif key.char == 'u':
                # Request uploading all saved episodes to the Hub
                print("\n↺ Upload request received: will push saved episodes when safe.")
                self.push_requested = True
                return

        except AttributeError:
            if key == keyboard.Key.esc:
                print("\nArrêt demandé par l'utilisateur.")
                if self._shutdown_event is not None:
                    self._shutdown_event.set()

    def run(self, ctx):
        engine = self._engine
        interpolator = self._interpolator
        step_duration = 1.0 / ctx.runtime.cfg.fps
        start_time = time.perf_counter()

        engine.pause()
        
        while not ctx.runtime.shutdown_event.is_set():
            loop_start = time.perf_counter()

            if ctx.runtime.cfg.duration > 0 and (time.perf_counter() - start_time) >= ctx.runtime.cfg.duration:
                break

            # 1. Lire les caméras et la position
            observation = ctx.hardware.robot_wrapper.get_observation()

            # 2. Machine à états
            if self.state == "WAITING":
                engine.pause()
                home_action = ctx.processors.robot_action_processor((self.home_action, observation))
                ctx.hardware.robot_wrapper.send_action(home_action)
                # If user requested an upload, perform it from the main thread (safe)
                if self.push_requested and self._dataset is not None:
                    try:
                        print("Uploading saved episodes to the Hub...")
                        safe_push_to_hub(
                            self._dataset,
                            tags=ctx.runtime.cfg.dataset.tags if ctx.runtime.cfg.dataset else None,
                            private=ctx.runtime.cfg.dataset.private if ctx.runtime.cfg.dataset else None,
                        )
                        print("Upload complete.")
                    except Exception as e:
                        print(f"Upload failed: {e}")
                    finally:
                        self.push_requested = False
                
            elif self.state == "RECORDING":
                engine.resume()
                obs_processed = self._process_observation_and_notify(ctx.processors, observation)

                if self._handle_warmup(ctx.runtime.cfg.use_torch_compile, loop_start, step_duration):
                    continue

                action_dict = send_next_action(obs_processed, observation, ctx, interpolator)
                if self._dataset is not None and action_dict is not None:
                    obs_frame = build_dataset_frame(ctx.data.dataset_features, obs_processed, prefix=OBS_STR)
                    action_frame = build_dataset_frame(ctx.data.dataset_features, action_dict, prefix=ACTION)
                    task_str = ctx.runtime.cfg.dataset.single_task if ctx.runtime.cfg.dataset else ctx.runtime.cfg.task
                    self._dataset.add_frame({**obs_frame, **action_frame, "task": task_str})
                    try:
                        log_rerun_data(observation=obs_processed, action=action_dict, compress_images=True)
                    except Exception:
                        # Visualization should not break the rollout loop
                        pass

            elif self.state == "RESETTING":
                engine.pause()
                home_action = ctx.processors.robot_action_processor((self.home_action, observation))
                ctx.hardware.robot_wrapper.send_action(home_action)

                if self.reset_started_at is not None and (time.perf_counter() - self.reset_started_at) >= self.reset_hold_seconds:
                    engine.reset()
                    interpolator.reset()
                    self.reset_started_at = None
                    if self.rerecord_requested:
                        if self._dataset is not None:
                            self._dataset.clear_episode_buffer()
                        self.rerecord_requested = False
                        self.state = "RECORDING"
                        print("✅ Épisode prêt. Enregistrement relancé.")
                    else:
                        # Save the just-finished episode (do NOT push immediately)
                        if self._dataset is not None:
                            try:
                                self._dataset.save_episode()
                                print(f"Épisode #{self.episode_index} sauvegardé localement.")
                            except Exception:
                                pass

                        self.state = "WAITING"
                        print("✅ Prêt. Appuyez sur 'D' pour le prochain essai.")

            # 3. Maintien de la fréquence (FPS)
            elapsed = time.perf_counter() - loop_start
            sleep_time = step_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def teardown(self, ctx):
        print("\nNettoyage et arrêt...")
        # Finalize and optionally push all saved episodes to the Hub
        if self._dataset is not None:
            try:
                # Ensure last episode is saved if it contains frames
                try:
                    self._dataset.save_episode()
                    if self.episode_index > 0:
                        print(f"Épisode #{self.episode_index} sauvegardé au teardown.")
                except Exception:
                    pass
                self._dataset.finalize()
            except Exception:
                pass

            if ctx.runtime.cfg.dataset and ctx.runtime.cfg.dataset.push_to_hub:
                try:
                    print("Pushing all saved episodes to the Hub...")
                    safe_push_to_hub(
                        self._dataset,
                        tags=ctx.runtime.cfg.dataset.tags,
                        private=ctx.runtime.cfg.dataset.private,
                    )
                    print("Push complete.")
                except Exception as e:
                    print(f"Push failed during teardown: {e}")
        if self.listener is not None:
            self.listener.stop()
        super().teardown(ctx)


# --- 2. Lancement Principal ---

def main():
    init_logging()

    # 1. Configuration du Robot
    # On utilise directement la classe Config importée de ton module custom
    my_robot_config = SO101FollowerDragontactileConfig(
        port=MY_ROBOT_PORT,
        cameras={
            "top": OpenCVCameraConfig(
                index_or_path=0,
                width=640,
                height=480,
                fps=FPS
            ),
            "wrist": OpenCVCameraConfig(
                index_or_path=2,
                width=640,
                height=480,
                fps=FPS
            )
        }
    )

    # 2. Configuration de la Politique
    my_policy_config = PreTrainedConfig.from_pretrained(HF_MODEL_ID)
    my_policy_config.pretrained_path = HF_MODEL_ID

    my_dataset_config = DatasetRecordConfig(
        repo_id=f"{HF_USERNAME}/rollout_{DATASET_NAME}",
        single_task="shake the metal box and place it depending on its contents",
        push_to_hub=True,
    )
    
    # 3. Configuration du Rollout
    cfg = RolloutConfig(
        robot=my_robot_config,
        policy=my_policy_config,
        strategy=SentryStrategyConfig(upload_every_n_episodes=1),
        dataset=my_dataset_config,
        inference=SyncInferenceConfig(),
        duration=0, # 0 = infini, contrôlé manuellement par la boucle "run"
        task="shake the metal box and place it depending on its contents",
        display_data=True,
    )

    signal_handler = ProcessSignalHandler(use_threads=True)
    
    # Le contexte compile tout (chargement des poids PyTorch, connexion au robot, threads caméras)
    ctx = build_rollout_context(
        cfg,
        signal_handler.shutdown_event
    )

    # 4. Lancement de la stratégie personnalisée
    strategy = ControlledStrategy(cfg.strategy)

    # lancement de la visualisation Rerun (optionnelle, mais recommandée pour le debug et la démo)
    if cfg.display_data:
        print("Initialisation de la visualisation Rerun...")
        init_rerun(session_name=f"rollout_{DATASET_NAME}", ip=cfg.display_ip, port=cfg.display_port)

    try:
        strategy.setup(ctx)
        strategy.run(ctx)
    except KeyboardInterrupt:
        pass
    finally:
        strategy.teardown(ctx)


if __name__ == "__main__":
    main()