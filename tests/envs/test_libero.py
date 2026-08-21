from lerobot.envs.libero import LiberoEnv


def test_close_clears_underlying_env_for_later_lazy_recreation():
    class FakeOffScreenRenderEnv:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    env = LiberoEnv.__new__(LiberoEnv)
    underlying = FakeOffScreenRenderEnv()
    env._env = underlying

    env.close()
    env.close()

    assert underlying.close_calls == 1
    assert env._env is None
