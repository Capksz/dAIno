import asyncio
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from playwright.async_api import async_playwright
import time

class DinoEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.browser = None
        self.page = None
        self.context = None
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._init_browser())

        self.last_action_time = time.time()

        self.action_space = spaces.Discrete(3)  # Noop, Jump, Duck
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([150, 600, 150, 150, 150, 600, 150, 150, 150, 600, 150, 150], dtype=np.float32),
            dtype=np.float32
        )

    async def _init_browser(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=False)
        self.context = await self.browser.new_context(viewport={"width": 1200, "height": 600})
        self.page = await self.context.new_page()
        await self.page.goto("https://trex-runner.com/")
        await self.page.keyboard.press("Space")

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.loop.run_until_complete(self._restart())
        return self._get_obs(), {}

    async def _restart(self):
        await self.page.evaluate("Runner.instance_.restart()")
        await self.page.keyboard.press("Space")

    def step(self, action):
        # print(action)
        self._send_action(action)
        obs = self._get_obs()
        done = self._get_crashed()
        reward = -100 if done else 1
        time.sleep(0.2)
        return obs, reward, done, False, {}

    async def _hold_key(self, key: str, duration: float):
        await self.page.keyboard.down(key)
        await asyncio.sleep(duration)
        await self.page.keyboard.up(key)

    def _send_action(self, action):
        now = time.time()
        if now - self.last_action_time < 0.15:
            return
        if action == 1:
            self.loop.run_until_complete(self.page.keyboard.press("Space"))
        elif action == 2:
            self.loop.run_until_complete(self._hold_key("ArrowDown", 0.2))
        self.last_action_time = now

    def _get_obs(self):
        data = self.loop.run_until_complete(
            self.page.evaluate("""
                (() => {
                    const trex = Runner.instance_.tRex;
                    const obs = Runner.instance_.horizon.obstacles;
                    const width = trex.ducking ? trex.config.WIDTH_DUCK : trex.config.WIDTH;
                    const height = trex.ducking ? trex.config.HEIGHT_DUCK : trex.config.HEIGHT;

                    const sortedObs = obs.slice().sort((a, b) => a.xPos - b.xPos);
                    const relevantObs = sortedObs.filter(o => o.xPos + o.width >= trex.xPos);

                    const getBox = (o) => o ? [o.xPos, o.yPos, o.width, o.typeConfig.height] : [600, 0, 0, 0];

                    return [
                        trex.xPos, trex.yPos, width, height,
                        ...getBox(relevantObs[0]),
                        ...getBox(relevantObs[1])
                    ];
                })();
            """)
        )
        return np.array(data, dtype=np.float32)

    def _get_crashed(self):
        return self.loop.run_until_complete(
            self.page.evaluate("Runner.instance_.crashed")
        )

    def close(self):
        self.loop.run_until_complete(self._shutdown())

    async def _shutdown(self):
        await self.context.close()
        await self.browser.close()
        await self.playwright.stop()
