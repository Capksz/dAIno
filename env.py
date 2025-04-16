import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DinoEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.driver = self._init_browser()
        self._start_game()

        self.action_space = spaces.Discrete(3)
        # Observation: [trex_y, obstacle_x, obstacle_y, obstacle_width]
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0]),
                                            high=np.array([150, 600, 150, 50]),
                                            dtype=np.float32)

    def _init_browser(self):
        chrome_options = Options()
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--window-size=1200x600")
        chrome_options.add_argument("--disable-dev-shm-usage")
        # chrome_options.add_argument("--headless")

        driver = webdriver.Chrome(options=chrome_options)
        driver.get("https://trex-runner.com/")

        time.sleep(2)
        return driver

    def _start_game(self):
        self.driver.execute_script("""
                document.dispatchEvent(new KeyboardEvent('keydown', {keyCode: 32}));
                document.dispatchEvent(new KeyboardEvent('keyup', {keyCode: 32}));
            """)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.driver.refresh()
        time.sleep(1.5)
        self._start_game()
        return self._get_game_state()["obs"], {}

    def step(self, action):
        self._send_action(action)
        time.sleep(0.1)
        state = self._get_game_state()
        obs = state["obs"]
        done = state["crashed"]

        if done:
            reward = -10000
        else:
            if action == 0:
                reward = 2.0
            else:
                reward = 1.0
        print(action, reward)
        terminated = done
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def _get_game_state(self):
        try:
            distance = self.driver.execute_script("return Runner.instance_.distanceRan")
            crashed = self.driver.execute_script("return Runner.instance_.crashed")
            trex_y = self.driver.execute_script("return Runner.instance_.tRex.yPos")
            trex_jump = self.driver.execute_script("return Runner.instance_.tRex.jumping")
            trex_duck = self.driver.execute_script("return Runner.instance_.tRex.ducking")
            trex_jump_velocity = self.driver.execute_script("return Runner.instance_.tRex.jumpVelocity")

            obstacles = self.driver.execute_script("""
                const obs = Runner.instance_.horizon.obstacles;
                if (obs.length > 0) {
                    return obs.map(o => ({
                        xPos: o.xPos,
                        yPos: o.yPos,
                        width: o.width,
                        height: o.typeConfig.height,
                        type: o.typeConfig.type
                    }));
                }
                return [];
            """)

            if obstacles:
                first = obstacles[0]
                obs_array = np.array([trex_y, first['xPos'], first['yPos'], first['width']], dtype=np.float32)
            else:
                obs_array = np.array([trex_y, 600, 0, 0], dtype=np.float32)
            print(obstacles)
            return {"obs": obs_array, "crashed": crashed}
        except Exception as e:
            print("JS read error:", e)
            return {"obs": np.zeros(4), "crashed": True}

    def _send_action(self, action):
        if action == 0:
            pass

        elif action == 1:  # jump
            self.driver.execute_script("""
                document.dispatchEvent(new KeyboardEvent('keydown', {keyCode: 32}));
                document.dispatchEvent(new KeyboardEvent('keyup', {keyCode: 32}));
                """)

        elif action == 2:
            self.driver.execute_script("""
                document.dispatchEvent(new KeyboardEvent('keydown', {keyCode: 40}));
                document.dispatchEvent(new KeyboardEvent('keyup', {keyCode: 40}));
                """)

    def close(self):
        self.driver.quit()
