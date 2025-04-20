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

        self.last_distance = 0

        self.action_space = spaces.Discrete(3)
        # Observation: [trex_x, trex_y, trex_width, trex_height, obs1_x, obs1_y, obs1_width, obs1_height, obs2_x, obs2_y, obs2_width, obs2_height]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([150, 600, 150, 150, 150, 600, 150, 150, 150, 600, 150, 150], dtype=np.float32),
            dtype=np.float32
        )

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

        self.driver.execute_script("Runner.instance_.restart()")
        time.sleep(0.05)
        self.last_distance = self.driver.execute_script("return Runner.instance_.distanceRan")
        return self._get_game_state()["obs"], {}

    def step(self, action):
        self._send_action(action)
        time.sleep(0.1)

        state = self._get_game_state()
        obs = state["obs"]
        done = state["crashed"]
        distance_now = state["distance"]

        reward = distance_now - self.last_distance
        self.last_distance = distance_now

        if done:
            reward = -10

        print(action, reward)
        return obs, reward, done, False, {}

    def _get_game_state(self):
        try:
            distance = self.driver.execute_script("return Runner.instance_.distanceRan")
            crashed = self.driver.execute_script("return Runner.instance_.crashed")
            trex_x = self.driver.execute_script("return Runner.instance_.tRex.xPos")
            trex_y = self.driver.execute_script("return Runner.instance_.tRex.yPos")
            width = self.driver.execute_script("return Runner.instance_.tRex.config.WIDTH")
            height = self.driver.execute_script("return Runner.instance_.tRex.config.HEIGHT")
            ducking = self.driver.execute_script("return Runner.instance_.tRex.ducking")
            duck_width = self.driver.execute_script("return Runner.instance_.tRex.config.WIDTH_DUCK")
            duck_height = self.driver.execute_script("return Runner.instance_.tRex.config.HEIGHT_DUCK")
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

            if trex_duck:
                width = duck_width
                height = duck_height

                # Sort obstacles by xPos (left boundary)
            obstacles.sort(key=lambda o: o['xPos'])

            # Skip all obstacles that are fully behind the Dino
            relevant_obstacles = [
                obs for obs in obstacles
                if obs['xPos'] + obs['width'] >= trex_x  # if obstacle not fully passed
            ]

            # Get up to 2 relevant obstacles
            obs1 = relevant_obstacles[0] if len(relevant_obstacles) > 0 else {"xPos": 600, "yPos": 0, "width": 0,
                                                                              "height": 0}
            obs2 = relevant_obstacles[1] if len(relevant_obstacles) > 1 else {"xPos": 600, "yPos": 0, "width": 0,
                                                                              "height": 0}

            obs_array = np.array([
                trex_x, trex_y, width, height,
                obs1['xPos'], obs1['yPos'], obs1['width'], obs1['height'],
                obs2['xPos'], obs2['yPos'], obs2['width'], obs2['height']
            ], dtype=np.float32)

            return {"obs": obs_array, "crashed": crashed, "distance": distance}

        except Exception as e:
            print("JS read error:", e)
            return {"obs": np.zeros(12, dtype=np.float32), "crashed": True}

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
